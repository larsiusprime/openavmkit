

## Configuring cloud storage

`openavmkit` includes a module for working with remote storage services. At this time the library supports three cloud storage methods:

- Microsoft Azure
- Hugging Face
- SFTP

To configure cloud storage, you will need to create a file that stores your connection credentials (such as API keys or passwords). This file should be named `.env` and should be placed in the `notebooks/` directory.

This file is already ignored by git, but do make sure you don't accidentally commit this file to the repository or share it with others, as it contains your sensitive login information!

This file should be a plain text file formatted like this:
```
SOME_VARIABLE=some_value
ANOTHER_VARIABLE=another_value
YET_ANOTHER_VARIABLE=123
```

That's just an example of the format; here are the actual variables that it recognizes:

| Variable Name                     | Description                                                                                                                                                             |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CLOUD_TYPE`                      | The type of cloud storage to use.<br>Legal values are: `azure`, `huggingface`, `sftp`. You can also set this value per-project in your `settings.json` as `cloud.type`. |
| `AZURE_ACCESS`                    | The type of access your azure account has.<br>Legal values are: `read_only`, `read_write`.                                                                              |
| `AZURE_STORAGE_CONTAINER_NAME`    | The name of the Azure storage container                                                                                                                                 |
| `AZURE_STORAGE_CONNECTION_STRING` | The connection string for the Azure storage account                                                                                                                     |
| `HF_ACCESS`                       | The type of access your huggingface account has.<br>Legal values are: `read_only`, `read_write`.                                                                        |
| `HF_TOKEN`                        | The Hugging Face API token                                                                                                                                              |
| `HF_REPO_ID`                      | The Hugging Face repository ID                                                                                                                                          |
| `SFTP_ACCESS`                     | The type of access your SFTP account has.<br>Legal values are: `read_only`, `read_write`.                                                                               |
| `SFTP_HOST`                       | The hostname of the SFTP server                                                                                                                                         |
| `SFTP_USERNAME`                   | The username for the SFTP server                                                                                                                                        |
| `SFTP_PASSWORD`                   | The password for the SFTP server                                                                                                                                        |
| `SFTP_PORT`                       | The port number for the SFTP server                                                                                                                                     |

You only need to provide values for the service that you're actually using. For instance, here's what the file might look like if you are using Hugging Face:

```
CLOUD_TYPE=huggingface
HF_ACCESS=read_write
HF_REPO_ID=landeconomics/localities-public
HF_TOKEN=<YOUR_HUGGING_FACE_API_TOKEN>
```

If you're just getting started, you can just use read-only access to an existing public repository. Here's an example of how to access the public datasets provided by the [The Center for Land Economics](https://landeconomics.org):

```
CLOUD_TYPE=huggingface
HF_ACCESS=read_only
HF_REPO_ID=landeconomics/localities
```

This will let you download the inputs for any of the Center for Land Economics' public datasets. Note that you will be unable to upload your changes and outputs to repositories that you have read-only access to.

If you want to sync with your own cloud storage, you will need to set up your own hosting account and then provide the appropriate credentials in the `.env` file.

If you have multiple projects stored on different cloud services, you can set the `CLOUD_TYPE` and `CLOUD_ACCESS` variables in your settings.json. This will allow you to switch between cloud services on a per-project basis. **Do not ever store credentials in your settings.json, however, as these are uploaded to the cloud!**