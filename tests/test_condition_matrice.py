import numpy as np

from helper import condition_matrice


class TestConditionMatrice(object):

    z = np.eye(10)[np.random.randint(0, 9, 100)]

    def test_condition_generator(self):
        y_gz, _ = condition_matrice(self.z)

        assert np.allclose(y_gz[:, 0, 0, :], self.z)

    def test_condition_discriminator(self):
        _, y_dx = condition_matrice(self.z)
        row, col = np.where(self.z == 1)

        for r, c in zip(row, col):
            assert (y_dx[r, :, :, c] == 1).all()
