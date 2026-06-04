import os

base_request_url = f"https://{os.getenv("POP_TEST_DOMAIN")}/request-management"
base_settings_url = f"https://{os.getenv("POP_TEST_DOMAIN")}/pop-requests/settings"

if __name__ == "__main__":
    pass
