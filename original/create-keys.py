import tenseal as ts
import base64

def write_data(file_name, data):
    if type(data) == bytes:
        # bytes to base64
        data = base64.b64encode(data)

    with open(file_name, 'wb') as f:
        f.write(data)

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2 ** 40
write_data("public.txt", context.serialize(save_public_key=True))
write_data("secret.txt", context.serialize(save_secret_key=True))