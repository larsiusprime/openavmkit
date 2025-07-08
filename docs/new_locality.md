## Creating a new locality

OpenAVMKit operates on the concept of a "locality", which is a geographic area that contains a set of properties. This can represent a city, a county, a neighborhood, or any other region or jurisdiction you want to analyze. To set one up, create a folder like this within openavmkit's `notebooks/` directory:

```
data/<locality_slug>/
```

Where `<locality_slug>` is a unique identifying name for your locality in a particularly opinionated format. That format is:

```
<country_code>-<state_or_province_code>-<locality_name>
```

- **Country code**: The 2-letter country code according to the [ISO 3166-1 standard](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2). For example, the country code for the United States is `us`, and the country code for Norway is `no`.

- **State/province code**: The 2-letter state or province code according to the [ISO 3166-2 standard](https://en.wikipedia.org/wiki/ISO_3166-2). For example, the state code for Texas is `tx`, and the state code for California is `ca`.

- **Locality name**: A human-readable name for the locality itself. This follows no particular standard and is entirely up to you.

The slug itself should be all lowercase and contain no spaces or special characters other than underscores.

Some examples:

```
us-nc-guilford    # Guilford County, North Carolina, USA
us-tx-austin      # City of Austin, Texas, USA
no-03-oslo        # City of Oslo, Norway
no-50-orkdal      # Orkdal kommune (county), Norway
```

Once you have your locality set up, you will want to set it up like this (using `us-nc-guilford` as an example):

```
data/
├──us-nc-guilford/
    ├── in/
    ├── out/
    ├── settings.json
```

The `in/` directory is where you will put your raw data files.   
The `out/` directory is where the output files will be saved.  
The `settings.json` file will drive all your modeling and analysis decisions for the library. For now you can just put a blank `{}` in there so that it will load, but you will want to consult the documentation / tutorials for how to construct this file.