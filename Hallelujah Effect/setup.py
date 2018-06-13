from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='hallelujah-effect',
    version='0.1',
    author = 'Brennon Bortz',
    author_email = 'brennon@brennonbortz.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Hallelujah Effect Models',
    requires=[]
)
