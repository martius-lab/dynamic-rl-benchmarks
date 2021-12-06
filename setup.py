import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(name="dyn_rl_benchmarks",
      version="1.0.3",
      description="Reinforcement learning benchmark problems set in dynamic environments.",
      author="Nico Gürtler",
      author_email="nico.guertler@tuebingen.mpg.de",
      long_description=README,
      long_description_content_type="text/markdown",
      license="MIT",
      url="https://github.com/martius-lab/dynamic-rl-benchmarks",
      keywords=["reinforcement learning", "reinforcement learning environmnts"],
      packages=find_packages(),
      package_data={'dyn_rl_benchmarks.envs.assets': ['drawbridge.obj', 'drawbridge.mtl']},
      install_requires=["numpy", "gym", "roboball2d", "wavefront_reader"]
)
