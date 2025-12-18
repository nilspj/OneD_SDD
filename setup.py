from setuptools import find_packages, setup

setup(
    name='OneD_SDD',
    packages=find_packages(include=['OneD_SDD']),
    package_data={'OneD_SDD': ['quadrature_points/lebedev_053.txt']},
    version='0.1.0',
    install_requires=["numpy"],
    description='One dimensional spin drift-diffusion solver',
    author='Nils Petter Joerstad',
    author_email='nils.jorstad@proton.me'
)