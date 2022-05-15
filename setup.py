import setuptools

setuptools.setup(name='bmodel',
                 version='1.0.0',
                 packages=["bmodel"],
                 install_requires=[
                     "numpy",
                     "numba",
                     "pandas",
                     "scipy"
                 ]
                 )
