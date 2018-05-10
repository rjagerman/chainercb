from setuptools import setup

setup(
    name='chainercb',
    version='0.1.0',
    description='Neural Contextual Bandits using Chainer',
    url='https://github.com/rjagerman/chainercb',
    download_url = 'https://github.com/rjagerman/chainercb/archive/v0.1.0.tar.gz',
    author='Rolf Jagerman',
    author_email='rjagerman@gmail.com',
    license='MIT',
    packages=['chainercb',
              'chainercb.policies',
              'test',
              'test.policies'],
    install_requires=['numpy>=1.13.0',
                      'chainer>=3.0.0'],
    test_suite='nose.collector',
    tests_require=['nose']
)

