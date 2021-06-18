# Specifying your environment

Parts of lwa-tools scan a file in this directory called `.env` for environment
information. You should add this information if necessary. This file is not
included in the repository because its contents will be different for everyone.

There is however a [env.sample](./env.sample) file in this directory which you
can copy into a `.env` which will be git ignored and store your secrets.

The following information should be present in that file:

- `KNOWN_TRANSMITTER_FILE` : the path of a file containing known transmitter information. It should be in CSV format with columns `name`, `lat`, `lon`.

- `OBSERVATION_RECORDSET_FILE` : the path of a file containing
observation metadata. It should be in csv format.