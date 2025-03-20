import setuptools

# Read the long description from the README.md file
with open("README.md","r",encoding = "utf-8") as f:
    long_description = f.read()

__version__ = '0.0.0'

# Define project metadata
REPO_NAME = "Breast Cancer Classification project"
AUTHOR_USER_NAME = "ADITY-STAR"
SOURCE_REPO = "Breast Cancer Classifier"
AUTHOR_EMAIL = "aakuskar.980@gmail.com"
DESCRIPTION = "A Python package for breast cancer classification using machine learning."


# Set up the package
setuptools.setup(
    name=SOURCE_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,  
    long_description=long_description,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},  
    packages=setuptools.find_packages(where="src"),
)