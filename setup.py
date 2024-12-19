from setuptools import setup, find_packages

setup(
    name='PATE',
    version='0.1.1',
    author='Ramin Ghorbani',
    author_email='r.ghorbani@tudelft.nl',
    packages=find_packages(include=['pate', 'pate.*']),  # Include only the 'pate' package
    description='PATE: Proximity-Aware Time series anomaly Evaluation metric',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Raminghorbanii/PATE',
    install_requires=open('requirements_pate.txt').read().splitlines(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',
)
