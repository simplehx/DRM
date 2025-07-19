start_token = "<"
mask_token = ">"
geo_code_type = "geohash"
if geo_code_type == "geohash":
    geo_code_char = "0123456789bcdefghjkmnpqrstuvwxyz" + start_token + mask_token
if geo_code_type == "h3":
    geo_code_char = "0123456789abcdefghijklmnopqrstuvwxyz" + start_token + mask_token
hash2index = {character: index for index, character in enumerate(geo_code_char)}
index2hash = {index:character for character, index in hash2index.items()}
K_list = [10, 50, 100]

sos_token_id = geo_code_char.find(start_token)