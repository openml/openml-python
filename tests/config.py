USE_STAGING = False

if not USE_STAGING:
    TEST_SUBDOMAIN = "test"
    TEST_SERVER = "https://test.openml.org/api/v1/xml"
    TEST_SERVER_API_KEY = "c0c42819af31e706efe1f4b88c23c6c1"
    TEST_SERVER_API_KEY_ADMIN = "610344db6388d9ba34f6db45a3cf71de"
    # amueller's read/write key that he will throw away "later"
else:
    TEST_SUBDOMAIN = "staging"
    TEST_SERVER = "https://staging.openml.org/api/v1/xml"
    TEST_SERVER_API_KEY = "normaluser"
    TEST_SERVER_API_KEY_ADMIN = "abc"
