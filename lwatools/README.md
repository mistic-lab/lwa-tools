# Specifying your environment

Parts of lwa-tools scan a file in this directory called `.env` for environment
information. You should add this information if necessary. This file is not
included in the repository because its contents will be different for everyone.

The following information should be present in that file:

`KNOWN_TRANSMITTER_FILE` : the path of a file containing known transmitter
information. It should be in CSV format with columns `name`, `lat`, `lon`.
