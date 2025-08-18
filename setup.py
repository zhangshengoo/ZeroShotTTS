from setuptools import setup, find_packages

setup(
    name="fishspeech-infer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click",
        "loguru",
        "torch",
        "torchaudio",
        "numpy",
        "hydra-core",
        "omegaconf",
        "transformers",
        "soundfile",
    ],
    entry_points={
        'console_scripts': [
            'fishspeech-infer=infer_api.fishspeech.infer_api:main',
            'fishspeech-text2semantic=infer_api.fishspeech.infer_text2semantic:main',
            'fishspeech-vqgan=infer_api.fishspeech.infer_vqgan:main',
        ],
    },
) 