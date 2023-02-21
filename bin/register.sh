#!/usr/bin/env bash

# see https://fink-broker.readthedocs.io/en/latest/topics/

fink_client_register \
  -username "tarek" \
  -group_id "tarek_fink" \
  -mytopics fink_early_sn_candidates_ztf \
  fink_sn_candidates_ztf \
  fink_sso_ztf_candidates_ztf \
  fink_sso_fink_candidates_ztf \
  fink_kn_candidates_ztf \
  fink_early_kn_candidates_ztf \
  fink_rate_based_kn_candidates_ztf \
  fink_microlensing_candidates_ztf \
  fink_simbad_ztf \
  -servers "134.158.74.95:24499" \
  -maxtimeout 5 \
  --verbose
