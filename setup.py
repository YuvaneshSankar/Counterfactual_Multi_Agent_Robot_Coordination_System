
from setuptools import setup, find_packages
import os

# Read the README file for long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith('#')
    ]

setup(
    name='warehouse-coma-marl',

    version='1.0.0',

    description='Counterfactual_Multi_Agent_Robot_Coordination_System',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/YuvaneshSankar/Counterfactual_Multi_Agent_Robot_Coordination_System',

    author='Yuvanesh Sankar',

    author_email='yuvanesh.skv@example.com',

    license='MIT',

    # Classifiers help users find your project by categorizing it
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    keywords=[
        'reinforcement-learning',
        'multi-agent',
        'COMA',
        'robotics',
        'warehouse-automation',
        'continuous-control',
        'pytorch',
        'deep-learning'
    ],

    # Package discovery
    packages=find_packages(exclude=['tests', 'notebooks', 'docs', 'assets']),

    # Minimum Python version
    python_requires='>=3.8',

    # Dependencies
    install_requires=requirements,

    # Optional dependencies for development and testing
    extras_require={
        'dev': [
            'black',
            'flake8',
            'mypy',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
        ],
        'jupyter': [
            'jupyter',
            'jupyterlab',
            'ipython',
        ],
        'robotics': [
            # Uncomment for real robot integration
            # 'rclpy>=0.13.0',
            # 'geometry-msgs>=0.0.12',
        ],
    },

    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'warehouse-train=scripts.train:main',
            'warehouse-evaluate=scripts.evaluate:main',
            'warehouse-visualize=scripts.visualize:main',
            'warehouse-benchmark=scripts.benchmark:main',
        ],
    },

    # Include additional package data
    include_package_data=True,

    package_data={
        'warehouse_coma_marl': [
            'configs/*.yaml',
            'assets/**/*',
        ],
    },

    # PyPI upload metadata
    project_urls={
        'Bug Reports': 'https://github.com/YuvaneshSankar/Counterfactual_Multi_Agent_Robot_Coordination_System/issues',
        'Source': 'https://github.com/YuvaneshSankar/Counterfactual_Multi_Agent_Robot_Coordination_System'
    },

    zip_safe=False,
)
