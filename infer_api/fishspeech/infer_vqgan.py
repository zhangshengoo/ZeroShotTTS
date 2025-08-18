from pathlib import Path
import sys

import hydra
import numpy as np
import soundfile as sf
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
import click

# Import fish-speech modules
try:
    from fish_speech.utils.file import AUDIO_EXTENSIONS
except ImportError:
    fish_speech_dir = Path(__file__).parent.parent.parent / "fish-speech"
    if fish_speech_dir.exists():
        sys.path.append(str(fish_speech_dir))
        from fish_speech.utils.file import AUDIO_EXTENSIONS
    else:
        raise ImportError("Could not find fish-speech package. Please make sure it's installed or the path is correct.")

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


def load_model(config_name, checkpoint_path, device="cuda"):
    """
    加载VQGAN模型
    
    Args:
        config_name: 模型配置名称
        checkpoint_path: 模型检查点路径
        device: 运行设备
    
    Returns:
        model: 加载好的模型
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(
        checkpoint_path, map_location=device, mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)

    logger.info(f"Loaded model: {result}")
    return model


@torch.no_grad()
def vqgan_encoder(input_path, output_path, config_name, checkpoint_path, device="cuda"):
    """
    VQGAN编码器接口：将音频文件编码为索引数据
    
    Args:
        input_path: 输入音频文件路径
        output_path: 输出索引文件路径(.npy)
        config_name: 模型配置名称
        checkpoint_path: 模型检查点路径
        device: 运行设备
    
    Returns:
        indices: 编码后的索引数据
        sample_rate: 音频采样率
    """
    model = load_model(config_name, checkpoint_path, device=device)
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.suffix not in AUDIO_EXTENSIONS:
        raise ValueError(f"Input file must be an audio file, got: {input_path}")

    logger.info(f"Processing audio file: {input_path}")

    # Load audio
    audio, sr = torchaudio.load(str(input_path))
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    audio = torchaudio.functional.resample(
        audio, sr, model.spec_transform.sample_rate
    )

    audios = audio[None].to(device)
    logger.info(
        f"Loaded audio with {audios.shape[2] / model.spec_transform.sample_rate:.2f} seconds"
    )

    # VQ Encoder
    audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
    indices = model.encode(audios, audio_lengths)[0][0]

    logger.info(f"Generated indices of shape {indices.shape}")

    # Save indices
    if output_path:
        np.save(output_path, indices.cpu().numpy())
        logger.info(f"Saved indices to {output_path}")
    
    return indices, model.spec_transform.sample_rate


@torch.no_grad()
def vqgan_decoder(indices, output_path, config_name, checkpoint_path, device="cuda", indices_file=None):
    """
    VQGAN解码器接口：将索引数据解码为音频文件
    
    Args:
        indices: 索引数据（torch.Tensor或None）
        output_path: 输出音频文件路径
        config_name: 模型配置名称
        checkpoint_path: 模型检查点路径
        device: 运行设备
        indices_file: 索引文件路径(.npy)，当indices为None时使用
    
    Returns:
        audio_data: 重建的音频数据
        sample_rate: 音频采样率
    """
    model = load_model(config_name, checkpoint_path, device=device)
    output_path = Path(output_path) if output_path else None

    # 从文件加载索引或使用传入的索引
    if indices is None:
        if indices_file is None:
            raise ValueError("Either indices or indices_file must be provided")
        
        indices_file = Path(indices_file)
        if indices_file.suffix != ".npy":
            raise ValueError(f"Indices file must be a .npy file, got: {indices_file}")

        logger.info(f"Processing indices file: {indices_file}")
        indices = np.load(indices_file)
        indices = torch.from_numpy(indices).to(device).long()
    
    if indices.ndim != 2:
        raise ValueError(f"Expected 2D indices, got {indices.ndim}D")

    # Restore
    feature_lengths = torch.tensor([indices.shape[1]], device=device)
    fake_audios, _ = model.decode(
        indices=indices[None], feature_lengths=feature_lengths
    )
    audio_time = fake_audios.shape[-1] / model.spec_transform.sample_rate

    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # Get audio data
    audio_data = fake_audios[0, 0].float().cpu().numpy()
    
    # Save audio if output path is provided
    if output_path:
        sf.write(output_path, audio_data, model.spec_transform.sample_rate)
        logger.info(f"Saved audio to {output_path}")

    return audio_data, model.spec_transform.sample_rate


@click.command()
@click.option(
    "--input-path",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input audio file path"
)
@click.option(
    "--output-path",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output audio file path"
)
@click.option(
    "--config-name",
    default="firefly_gan_vq",
    help="Model configuration name"
)
@click.option(
    "--checkpoint-path",
    required=True,
    type=click.Path(exists=True),
    help="Model checkpoint path"
)
@click.option(
    "--test-mode",
    type=click.Choice(["file", "memory"]),
    default="file",
    help="Test mode: 'file' for saving intermediate files, 'memory' for direct processing"
)
def main(input_path, output_path, config_name, checkpoint_path, test_mode):
    """
    VQGAN编码解码测试程序
    
    支持两种测试模式：
    1. file模式：保存中间的索引文件
    2. memory模式：在内存中直接处理，不保存中间文件
    """
    logger.info(f"Running test in {test_mode} mode")
    
    if test_mode == "file":
        # 方式1：完整的编码-解码流程（保存中间文件）
        logger.info("Step 1: Encoding audio to indices")
        indices_path = Path(output_path).parent / "indices.npy"
        indices, sr = vqgan_encoder(
            input_path=input_path,
            output_path=indices_path,
            config_name=config_name,
            checkpoint_path=checkpoint_path
        )
        
        logger.info("Step 2: Decoding indices to audio")
        audio_data, sr = vqgan_decoder(
            indices=None,
            indices_file=indices_path,
            output_path=output_path,
            config_name=config_name,
            checkpoint_path=checkpoint_path
        )
        
    else:  # memory mode
        # 方式2：内存中直接处理（不保存中间文件）
        logger.info("Step 1: Encoding audio to indices (in memory)")
        indices, sr = vqgan_encoder(
            input_path=input_path,
            output_path=None,
            config_name=config_name,
            checkpoint_path=checkpoint_path
        )
        
        logger.info("Step 2: Decoding indices to audio (in memory)")
        audio_data, sr = vqgan_decoder(
            indices=indices,
            output_path=output_path,
            config_name=config_name,
            checkpoint_path=checkpoint_path
        )
    
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    main()
