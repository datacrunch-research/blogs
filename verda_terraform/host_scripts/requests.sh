#! /bin/bash

set -euo pipefail

BASE_URL="${BASE_URL:-https://api.datacrunch.io/v1}"
VERDA_CLIENT_ID=$VERDA_CLIENT_ID
VERDA_CLIENT_SECRET=$VERDA_CLIENT_SECRET

RESP=$(
  curl -sS --request POST "$BASE_URL/oauth2/token" \
    --header "Content-Type: application/json" \
    --data "{\"grant_type\":\"client_credentials\",\"client_id\":\"$VERDA_CLIENT_ID\",\"client_secret\":\"$VERDA_CLIENT_SECRET\"}"
)

#echo "Raw response:"
#echo "$RESP"

if command -v jq >/dev/null 2>&1; then
  TOKEN="$(echo "$RESP" | jq -r '.access_token')"
  echo
  #echo "access_token:"
  #echo "$TOKEN"
fi

echo
RESP=$(
  curl -sS --request GET "$BASE_URL/instance-types" \
      --header "Authorization: Bearer $TOKEN" \
    )

echo "instances types:"
echo "$RESP" | jq -r '.[].instance_type'

echo
RESP=$(
  curl -sS --request GET "$BASE_URL/images" \
      --header "Authorization: Bearer $TOKEN" \
    )

echo "images:"
echo "$RESP" | jq -r '.[].image_type'



echo