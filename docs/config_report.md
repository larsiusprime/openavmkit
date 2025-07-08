
## Configuring PDF report generation

`openavmkit` includes a module for generating PDF reports. This module uses the `pdfkit` library, which is a Python wrapper for the `wkhtmltopdf` command line tool. Although `pdfkit` will be installed automatically along with the rest of the dependencies, you will need to install `wkhtmltopdf` manually on your system to use this feature. If you skip this step, don't worry, you'll still be able to use the rest of the library, it just won't generate PDF reports.

### Installing wkhtmltopdf:

#### Manual installation

Visit the [wkhtmltopdf download page](https://wkhtmltopdf.org/downloads.html) and download the appropriate installer for your operating system.

#### Windows

1. Download the installer linked above
2. Run the installer and follow the instructions
3. Add the `wkhtmltopdf` installation directory to your system's PATH environment variable

If you don't know what 3. means:

The idea is that you want the `wkhtmltopdf` executable to be available from any command prompt, so you can run it from anywhere. For that to work, you need to make sure that the folder that the `wkhtmltopdf` executable is in is listed in your system's PATH environment variable.

Here's how to do that:

1. Find the folder where `wkhtmltopdf` was installed. It's probably in `C:\Program Files\wkhtmltopdf\bin`, but it could be somewhere else. Pay attention when you install it.
2. Follow this [tutorial](https://web.archive.org/web/20250131024033/https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/) to edit your PATH environment variable. You want to add the folder from step 1 to the PATH variable.
3. Open a new command prompt and type `wkhtmltopdf --version`. If you see a version number, you're all set!

#### Linux

On Debian/Ubuntu, run:

```bash
sudo apt-get update
sudo apt-get install wkhtmltopdf
```

#### macOS

Ensure you have [Homebrew](https://brew.sh) installed. Then run:

```bash
brew install wkhtmltopdf
```