from setuptools import find_packages, setup


#REQUIREMENT= 'requirements.txt'
HYPEN_E_DOT= '-e .'

def get_requirements():
    requirements=[]

    with open('requirements.txt') as file_obj:
        requirements = file_obj.readlines()
       # requirements = [req.replace("\n") for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements    

setup(
    name='CementStrengrhPrediction',
    version='0.0.1',
    author='sunand',
    author_email='sunandd92@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)
