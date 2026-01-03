from __future__ import annotations

import requests

from openml.__version__ import __version__


class HTTPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {"user-agent": f"openml-python/{__version__}"}

    def get(self, path, params=None):
        url = f"{self.base_url}/{path}"
        return requests.get(url, params=params, headers=self.headers)

    def post(self, path, data=None, files=None):
        url = f"{self.base_url}/{path}"
        return requests.post(url, data=data, files=files, headers=self.headers)

    def delete(self, path, params=None):
        url = f"{self.base_url}/{path}"
        return requests.delete(url, params=params, headers=self.headers)
