import os
from pathlib import Path
from typing import Optional, Union, Tuple, List

import click
import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer

# Import fish-speech modules
try:
    from fish_speech.models.text2semantic.llama import BaseModelArgs, BaseTransformer
    from fish_speech.text import clean_text
except ImportError:
    import sys
    fish_speech_dir = Path(__file__).parent.parent.parent / "fish-speech"
    if fish_speech_dir.exists():
        sys.path.append(str(fish_speech_dir))
        from fish_speech.models.text2semantic.llama import BaseModelArgs, BaseTransformer
        from fish_speech.text import clean_text
    else:
        raise ImportError("Could not find fish-speech package. Please make sure it's installed or the path is correct.")


def load_model(checkpoint_path: Union[str, Path], device: str = "cuda", compile: bool = False, half: bool = False) -> Tuple[BaseTransformer, callable]:
    """
    加载text2semantic模型
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 运行设备
        compile: 是否编译模型
        half: 是否使用半精度
    
    Returns:
        model: 加载好的模型
        decode_one_token: 解码函数
    """
    from fish_speech.models.text2semantic.inference import (
        load_model as load_model_internal,
    )
    
    precision = torch.half if half else torch.bfloat16
    model, decode_one_token = load_model_internal(
        checkpoint_path, device, precision, compile=compile
    )
    
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    
    return model, decode_one_token


def text2semantic(
    text: str,
    prompt_text: Optional[str] = None,
    prompt_tokens: Optional[torch.Tensor] = None,
    checkpoint_path: Union[str, Path] = None,
    device: str = "cuda",
    compile: bool = False,
    half: bool = False,
    max_new_tokens: int = 0,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    temperature: float = 0.7,
    iterative_prompt: bool = True,
    chunk_length: int = 100,
    num_samples: int = 1,
    seed: int = 42,
) -> List[torch.Tensor]:
    """
    文本转语义token接口
    
    Args:
        text: 输入文本
        prompt_text: 参考文本
        prompt_tokens: 参考文本对应的token (从VQGAN编码器获得)
        checkpoint_path: 模型检查点路径
        device: 运行设备
        compile: 是否编译模型
        half: 是否使用半精度
        max_new_tokens: 最大生成token数
        top_p: top-p采样参数
        repetition_penalty: 重复惩罚参数
        temperature: 温度参数
        iterative_prompt: 是否使用迭代prompt
        chunk_length: 文本分块长度
        num_samples: 生成样本数量
        seed: 随机种子
    
    Returns:
        codes_list: 生成的语义token列表，每个元素对应一个样本
    """
    from fish_speech.models.text2semantic.inference import generate_long
    
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided")
    
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    model, decode_one_token = load_model(checkpoint_path, device, compile, half)
    
    # Setup model caches
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    if prompt_text is not None and prompt_tokens is not None:
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]
    
    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )
    
    codes_list = []  # List to store codes for each sample
    current_sample_codes = []  # Temporary list to store codes for current sample
    
    for response in generator:
        if response.action == "sample":
            current_sample_codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if current_sample_codes:  # If we have codes for current sample
                # Concatenate all codes for current sample along sequence length dimension
                sample_codes = torch.cat(current_sample_codes, dim=1)
                codes_list.append(sample_codes)
                current_sample_codes = []  # Reset for next sample
    
    if not codes_list:
        raise RuntimeError("No codes generated")
    
    return codes_list


@click.command()
@click.option(
    "--text",
    type=str,
    required=True,
    help="Input text to convert"
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for semantic tokens (.npy)"
)
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Model checkpoint path"
)
@click.option(
    "--prompt-text",
    type=str,
    default=None,
    help="Reference text for voice cloning"
)
@click.option(
    "--prompt-tokens",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Reference tokens file (.npy) for voice cloning"
)
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--half/--no-half", default=False)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.7)
@click.option("--repetition-penalty", type=float, default=1.2)
@click.option("--temperature", type=float, default=0.7)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=100)
@click.option(
    "--num-samples",
    type=int,
    default=1,
    help="Number of samples to generate"
)
@click.option("--seed", type=int, default=42)
def main(
    text: str,
    output_path: Path,
    checkpoint_path: Path,
    prompt_text: Optional[str] = None,
    prompt_tokens: Optional[Path] = None,
    device: str = "cuda",
    compile: bool = False,
    half: bool = False,
    max_new_tokens: int = 0,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    temperature: float = 0.7,
    iterative_prompt: bool = True,
    chunk_length: int = 100,
    num_samples: int = 1,
    seed: int = 42,
) -> None:
    """
    文本转语义token的命令行接口
    """
    # 加载prompt tokens
    prompt_tokens_tensor = None
    if prompt_tokens is not None:
        if prompt_text is None:
            raise ValueError("prompt_text must be provided when prompt_tokens is provided")
        prompt_tokens_tensor = torch.from_numpy(np.load(prompt_tokens)).to(device)
    
    # 生成语义token
    codes_list = text2semantic(
        text=text,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens_tensor,
        checkpoint_path=checkpoint_path,
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
        seed=seed,
    )
    
    # 保存结果
    if num_samples > 1:
        output_dir = output_path.parent
        stem = output_path.stem
        suffix = output_path.suffix
        for i in range(num_samples):
            output_file = output_dir / f"{stem}_{i}{suffix}"
            np.save(output_file, codes_list[i].cpu().numpy())
            logger.info(f"Saved semantic tokens to {output_file}")
    else:
        np.save(output_path, codes_list[0].cpu().numpy())
        logger.info(f"Saved semantic tokens to {output_path}")


if __name__ == "__main__":
    main()
