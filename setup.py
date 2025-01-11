from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT= '-e .'
def get_requirement(file_path:str)->List[str]:
    '''This function will return the list of libraries
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements




setup(
name="MRI-stenosis-classification",
version="0.0.1",
email="iloyadayvo@gmail.com",
author="Iloyadayvo",
packages=find_packages(),
install_requires= get_requirement('requirements.txt')


)