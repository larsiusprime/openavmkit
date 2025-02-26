The purpose of the utilities folder is to solve issues with circular imports. Here are the rules:

Modules in the utilities folder:
- May not import each other.
- May not import any other module in the openavmkit package.
- May import modules not from the openavmkit package.
