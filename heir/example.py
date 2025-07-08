from heir import compile
from heir.mlir import F32, I16, I64, Secret

@compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
def foo(x: Secret[I16], y: Secret[I16]):
    sum = x + y
    diff = x - y
    mul = x * y
    expression = sum * diff + mul
    return expression

if __name__ == "__main__":
    foo.setup()
    enc_a = foo.encrypt_x(2)
    enc_b = foo.encrypt_y(2)
    result_enc = foo.eval(enc_a, enc_b)
    result = foo.decrypt_result(result_enc)

    print(
      f"Expected result for `func`: {foo.original(2,2)}, FHE result:"
      f" {result}"
    )
