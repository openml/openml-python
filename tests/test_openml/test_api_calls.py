import openml
import openml.testing


class TestConfig(openml.testing.TestBase):

    def test_too_long_uri(self):
        with self.assertRaisesRegex(
            openml.exceptions.OpenMLServerError,
            'URI too long!',
        ):
            openml.datasets.list_datasets(data_id=list(range(10000)))
