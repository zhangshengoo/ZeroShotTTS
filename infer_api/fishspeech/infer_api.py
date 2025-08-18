import os
from pathlib import Path
from typing import Optional
import sys

import click
from loguru import logger

# Add parent directory to Python path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from .infer_vqgan import vqgan_encoder, vqgan_decoder
from .infer_text2semantic import text2semantic


@click.command()
@click.option(
    "--text",
    type=str,
    default="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
    help="Input text to convert to speech"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="output.wav",
    help="Output audio file path"
)
@click.option(
    "--reference-audio",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Reference audio file for voice cloning"
)
@click.option(
    "--reference-text",
    type=str,
    default=None,
    help="Text content of the reference audio"
)
@click.option(
    "--llama-checkpoint-path",
    type=click.Path(exists=True, path_type=Path),
    default="checkpoints/fish-speech-1.5",
    help="Path to the text2semantic model checkpoint"
)
@click.option(
    "--decoder-checkpoint-path",
    type=click.Path(exists=True, path_type=Path),
    default="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    help="Path to the VQGAN decoder checkpoint"
)
@click.option(
    "--decoder-config-name",
    type=str,
    default="firefly_gan_vq",
    help="VQGAN decoder configuration name"
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to run inference on"
)
@click.option(
    "--compile/--no-compile",
    default=False,
    help="Whether to compile the model for faster inference"
)
@click.option(
    "--half/--no-half",
    default=False,
    help="Whether to use half precision"
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=0,
    help="Maximum number of new tokens to generate"
)
@click.option(
    "--top-p",
    type=float,
    default=0.7,
    help="Top-p sampling parameter"
)
@click.option(
    "--repetition-penalty",
    type=float,
    default=1.2,
    help="Repetition penalty parameter"
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature parameter"
)
@click.option(
    "--save-intermediate/--no-save-intermediate",
    default=False,
    help="Whether to save intermediate files"
)
@click.option(
    "--chunk-length",
    type=int,
    default=100,
    help="Text chunk length for processing"
)
@click.option(
    "--iterative-prompt/--no-iterative-prompt",
    default=True,
    help="Whether to use iterative prompting"
)
@click.option(
    "--num-samples",
    type=int,
    default=1,
    help="Number of samples to generate"
)
def main(
    text: str,
    output: Path,
    reference_audio: Optional[Path],
    reference_text: Optional[str],
    llama_checkpoint_path: Path,
    decoder_checkpoint_path: Path,
    decoder_config_name: str,
    device: str,
    compile: bool,
    half: bool,
    max_new_tokens: int,
    top_p: float,
    repetition_penalty: float,
    temperature: float,
    save_intermediate: bool,
    chunk_length: int,
    iterative_prompt: bool,
    num_samples: int,
) -> None:
    """
    Fish-Speech 命令行推理接口
    
    完整的推理流程包括：
    1. (可选) 从参考音频生成prompt tokens
    2. 从文本生成语义tokens
    3. 从语义tokens生成语音
    """
    # 创建临时目录用于存储中间文件
    temp_dir = Path("temp")
    if save_intermediate:
        os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 步骤1: 处理参考音频 (如果有)
        prompt_tokens = None
        if reference_audio is not None:
            if reference_text is None:
                raise ValueError("reference_text must be provided when reference_audio is provided")
            
            logger.info("Step 1: Processing reference audio...")
            prompt_path = temp_dir / "prompt.npy" if save_intermediate else None
            prompt_tokens, _ = vqgan_encoder(
                input_path=reference_audio,
                output_path=prompt_path,
                config_name=decoder_config_name,
                checkpoint_path=decoder_checkpoint_path,
                device=device
            )
            
            if save_intermediate:
                logger.info(f"Saved prompt tokens to {prompt_path}")
        
        # 步骤2: 生成语义tokens
        logger.info("Step 2: Generating semantic tokens...")
        semantic_path = temp_dir / "semantic.npy" if save_intermediate else None
        semantic_tokens = text2semantic(
            text=text,
            prompt_text=reference_text,
            prompt_tokens=prompt_tokens,
            checkpoint_path=llama_checkpoint_path,
            device=device,
            compile=compile,
            half=half,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            iterative_prompt=iterative_prompt,
            chunk_length=chunk_length,
            num_samples=num_samples,
        )
        
        if save_intermediate:
            if num_samples > 1:
                for i in range(num_samples):
                    np.save(temp_dir / f"semantic_{i}.npy", semantic_tokens[i].cpu().numpy())
                    logger.info(f"Saved semantic tokens to {temp_dir}/semantic_{i}.npy")
            else:
                np.save(semantic_path, semantic_tokens.cpu().numpy())
                logger.info(f"Saved semantic tokens to {semantic_path}")
        
        # 步骤3: 生成语音
        logger.info("Step 3: Generating speech...")
        if num_samples > 1:
            for i in range(num_samples):
                output_path = output.parent / f"{output.stem}_{i}{output.suffix}"
                audio_data, sample_rate = vqgan_decoder(
                    indices=semantic_tokens[i],  # 使用完整的语义token序列
                    output_path=output_path,
                    config_name=decoder_config_name,
                    checkpoint_path=decoder_checkpoint_path,
                    device=device
                )
                logger.info(f"Successfully generated speech at {output_path}")
        else:
            audio_data, sample_rate = vqgan_decoder(
                indices=semantic_tokens[0],  # 使用完整的语义token序列
                output_path=output,
                config_name=decoder_config_name,
                checkpoint_path=decoder_checkpoint_path,
                device=device
            )
            logger.info(f"Successfully generated speech at {output}")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
