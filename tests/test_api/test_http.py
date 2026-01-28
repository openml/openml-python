from requests import Response, Request
import time
import xmltodict
from openml.testing import TestAPIBase


class TestHTTPClient(TestAPIBase):
    def test_cache(self):
        url = self._get_url(path="task/31")
        params = {"param1": "value1", "param2": "value2"}

        key = self.cache.get_key(url, params)

        # validate key
        self.assertEqual(
            key,
            "org/openml/test/api/v1/task/31/param1=value1&param2=value2",
        )

        # create fake response
        req = Request("GET", url).prepare()
        response = Response()
        response.status_code = 200
        response.url = url
        response.reason = "OK"
        response._content = b"<xml>test</xml>"
        response.headers = {"Content-Type": "text/xml"}
        response.encoding = "utf-8"
        response.request = req
        response.elapsed = type("Elapsed", (), {"total_seconds": lambda self: 0.1})()

        # save to cache
        self.cache.save(key, response)

        # load from cache
        cached_response = self.cache.load(key)

        # validate loaded response
        self.assertEqual(cached_response.status_code, 200)
        self.assertEqual(cached_response.url, url)
        self.assertEqual(cached_response.content, b"<xml>test</xml>")
        self.assertEqual(
            cached_response.headers["Content-Type"], "text/xml"
        )

    def test_get(self):
        response = self.http_client.get("task/1")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<oml:task", response.content)

    def test_get_with_cache_creates_cache(self):
        response = self.http_client.get("task/1", use_cache=True)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.cache.path.exists())

        # verify cache directory structure exists
        cache_key = self.cache.get_key(
            self._get_url(path="task/1"),
            {},
        )
        cache_path = self.cache._key_to_path(cache_key)

        self.assertTrue((cache_path / "meta.json").exists())
        self.assertTrue((cache_path / "headers.json").exists())
        self.assertTrue((cache_path / "body.bin").exists())

    def test_get_uses_cached_response(self):
        # first request populates cache
        response1 = self.http_client.get("task/1", use_cache=True)

        # second request should load from cache
        response2 = self.http_client.get("task/1", use_cache=True)

        self.assertEqual(response1.content, response2.content)
        self.assertEqual(response1.status_code, response2.status_code)

    def test_get_cache_expires(self):
        # force short TTL
        self.cache.ttl = 1
        path = "task/1"

        url = self._get_url(path=path)
        key = self.cache.get_key(url, {})
        cache_path = self.cache._key_to_path(key) / "meta.json"

        response1 = self.http_client.get(path, use_cache=True)
        response1_cache_time_stamp = cache_path.stat().st_ctime

        time.sleep(2)

        response2 = self.http_client.get(path, use_cache=True)
        response2_cache_time_stamp = cache_path.stat().st_ctime

        # cache expired -> new request
        self.assertNotEqual(response1_cache_time_stamp, response2_cache_time_stamp)
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response1.content, response2.content)

    def test_post_and_delete(self):
        task_xml = """
        <oml:task_inputs xmlns:oml="http://openml.org/openml">
            <oml:task_type_id>5</oml:task_type_id>
            <oml:input name="source_data">193</oml:input>
            <oml:input name="estimation_procedure">17</oml:input>
        </oml:task_inputs>
        """

        task_id = None
        try:
            # POST the task
            post_response = self.http_client.post(
                "task",
                files={"description": task_xml},
            )
            self.assertEqual(post_response.status_code, 200)
            xml_resp = xmltodict.parse(post_response.content)
            task_id = int(xml_resp["oml:upload_task"]["oml:id"])

            # GET the task to verify it exists
            get_response = self.http_client.get(f"task/{task_id}")
            self.assertEqual(get_response.status_code, 200)

        finally:
            # DELETE the task if it was created
            if task_id is not None:
                try:
                    del_response = self.http_client.delete(f"task/{task_id}")
                    # optional: verify delete
                    if del_response.status_code != 200:
                        print(f"Warning: delete failed for task {task_id}")
                except Exception as e:
                    print(f"Warning: failed to delete task {task_id}: {e}")
