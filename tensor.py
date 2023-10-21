import numpy as np
import torch
from damgard_jurik import PublicKey, PrivateKeyRing


class EncryptedTensor:
    @staticmethod
    def Create(tensor, public_key: PublicKey):
        tensor_np = np.array(tensor)
        as_list = (
            np.vectorize(lambda a: (a + 1) * 1e9)(tensor_np.flatten())
            .astype(int)
            .tolist()
        )

        encrypted_tensor = public_key.encrypt_list(as_list)

        return EncryptedTensor(tensor_np.shape, encrypted_tensor, 1)

    def __init__(self, shape, data, divisor):
        self.shape = shape
        self.data = data
        self.divisor = divisor

    def __add__(self, other):
        print(f"Addition of {self.shape} and {other.shape}")

        if not self._shape_equal(other):
            raise ValueError("Shapes must be equal for addition of tensors")

        new_data = []
        for i in range(len(self.data)):
            new_data.append(self.data[i] + other.data[i])

        return EncryptedTensor(self.shape, new_data, self.divisor + other.divisor)

    def _shape_equal(self, other):
        if len(self.shape) is not len(other.shape):
            return False
        for i in range(len(self.shape)):
            if self.shape[i] is not other.shape[i]:
                return False
        return True

    def to_tensor(self, private_key_ring: PrivateKeyRing):
        data = np.vectorize(lambda a: (float(a) / 1e9 / self.divisor) - 1)(
            np.array(private_key_ring.decrypt_list(self.data))
        )
        tensor_np = np.reshape(data, self.shape)
        return torch.from_numpy(tensor_np)
