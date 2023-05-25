from autofeat import SetupProperties
from distutils.core import setup

setup_properties = SetupProperties()

setup(
    name=setup_properties.get_name(),
    version=setup_properties.get_version(),
    description=setup_properties.get_description(),
    author=setup_properties.get_author(),
    author_email=setup_properties.get_author_email(),
    url=setup_properties.get_url(),
    packages=[p for p in setup_properties.get_packages()],
)
