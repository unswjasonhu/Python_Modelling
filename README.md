
### Hazewatch Estimates

Remember to add tmp directory to /var/www and change the permissions for theano and permission issues
Remember to move the .theanorc file to the /var/www folder. python files called using cgi are called using
the uses www-data. The $HOME for this is /var/www. When theano is imported it will create a compile directory.
We want it to use the tmp directory (specified in the theanorc) for the compilation.

To understand this it will be helpful to read the code in this order:
1. resources.py
2. populate_FixedSamples.py
3. populate_estimates.py
4. populate_model.py
5. train_model.py
