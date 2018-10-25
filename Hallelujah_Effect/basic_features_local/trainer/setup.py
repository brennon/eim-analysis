from setuptools import find_packages
from setuptools import setup

# REQUIRED_PACKAGES = ['tensorflow_transform==0.6.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=None,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)