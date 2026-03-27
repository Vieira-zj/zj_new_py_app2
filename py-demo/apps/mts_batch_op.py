# -- coding: utf-8 --

import json
import os
from typing import Final, Optional, TypedDict

import requests
from dotenv import load_dotenv
from requests.exceptions import HTTPError
from requests.models import Response

load_dotenv()


class Endpoint(TypedDict):
    id: int
    path: str


class GatewayConfig(TypedDict):
    endpoint_id: int
    environments: list[str]
    gateway_config: str
    selected_approvers: list[str]
    change_comment: str
    is_formily: bool


class MtsBatchOp:
    def __init__(self) -> None:
        self.env: Final[str] = "test"
        self.approver: Final[str] = "jin.zheng"

        self.pop_domain: str = os.getenv("POP_DOMAIN", "")
        self.sp_host: str = os.getenv("SP_HOST", "")
        self.sp_token: str = os.getenv("SP_TOKEN", "")
        if not all((self.pop_domain, self.sp_host, self.sp_token)):
            raise ValueError("pop_domain, sp_host, sp_token is empty")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "authorization": f"Bearer {self.sp_token}",
                "content-type": "application/json",
            }
        )

    def get_first_endpoint(self, keyword: str) -> Optional[Endpoint]:
        endpoints = self.search_endpoints(keyword)
        return endpoints[0] if len(endpoints) > 0 else None

    def search_endpoints(self, keyword: str) -> list[Endpoint]:
        if not keyword:
            return []

        url = self.sp_host + "/v1/ecp_nonlive/get_endpoints_v2"
        data = {
            "domain": self.pop_domain,
            "need_config_statuses": True,
            "path": keyword,
        }
        resp: Response = self.session.post(url, json=data)
        resp_data = self.check_and_get_resp_data(resp)

        endpoints: list[Endpoint] = []
        for ep in resp_data["endpoints"]:
            endpoints.append(Endpoint(id=ep["id"], path=ep["endpoint"]))
        return endpoints

    def get_endpoint_config(self, endpoint_id: int) -> dict:
        if endpoint_id < 1:
            raise ValueError(f"invalid endpoint_id: {endpoint_id}")

        url = f"{self.sp_host}/v1/ecp_nonlive/endpoint_config/get?endpoint_id={endpoint_id}&environment={self.env}&is_formily=false"
        resp: Response = self.session.get(url)
        resp_data = self.check_and_get_resp_data(resp)

        gateway_config = resp_data["gateway_config"]
        print(f"get gateway config for endpoint [{endpoint_id}]: {gateway_config}")

        gateway_config_obj = json.loads(gateway_config)
        if gateway_config_obj is None or len(gateway_config_obj["spex"]) == 0:
            raise ValueError(f"invalid gateway config: endpoint_id={endpoint_id}")
        return gateway_config_obj

    def update_enpoint(self, endpoint_path: str, **kwargs) -> None:
        ep = self.get_first_endpoint(endpoint_path)
        if ep and ep.get("id"):
            return self.update_enpoint_by_id(ep.get("id"), **kwargs)
        raise KeyError(f"no endpoint found by path: {endpoint_path}")

    def update_enpoint_by_id(self, endpoint_id: int, **kwargs) -> None:
        if not kwargs:
            return None

        # do update
        endpoint_config = self.get_endpoint_config(endpoint_id)
        for k, v in kwargs.items():
            print(f"update gateway config [{k}]: {endpoint_config[k]}->{v}")
            endpoint_config[k] = v

        data = GatewayConfig(
            endpoint_id=endpoint_id,
            environments=[self.env],
            gateway_config=json.dumps(endpoint_config),
            selected_approvers=[self.approver],
            change_comment="update config",
            is_formily=False,
        )

        # print("[debug] update endpoint request:", data)
        url = self.sp_host + "/v1/ecp_nonlive/endpoint_config/update_multi_envs"
        resp: Response = self.session.post(url, json=data)
        resp_data = self.check_and_get_resp_data(resp)
        print(
            f"update gateway config success: endpoint_id={endpoint_id}, response_data={resp_data}"
        )
        return None

    def copy_endpoint(self, endpoint_path: str, to_env: str) -> None:
        if not to_env:
            return None

        ep = self.get_first_endpoint(endpoint_path)
        if not ep or not ep.get("id"):
            raise KeyError(f"no endpoint found by path: {endpoint_path}")

        endpoint_config = self.get_endpoint_config(ep.get("id"))
        data = GatewayConfig(
            endpoint_id=ep.get("id"),
            environments=[to_env],
            gateway_config=json.dumps(endpoint_config),
            selected_approvers=[self.approver],
            change_comment=f"copy config to {to_env}",
            is_formily=False,
        )

        url = self.sp_host + "/v1/ecp_nonlive/endpoint_config/update_multi_envs"
        resp: Response = self.session.post(url, json=data)
        resp_data = self.check_and_get_resp_data(resp)
        print(
            f"copy gateway config success: endpoint_path={endpoint_path}, to_env={to_env} response_data={resp_data}"
        )
        return None

    def create_endpoint(self, from_endpoint_id: int):
        print(f"to impl: create endpoint from {from_endpoint_id}")

    def check_and_get_resp_data(self, resp: Response) -> dict:
        resp.raise_for_status()
        resp_json: dict = resp.json()
        if resp_json["error"]:
            raise HTTPError(
                f"http request failed: error={resp_json["error"]}, error_msg={resp_json["error_msg"]}"
            )
        return resp_json["data"]


if __name__ == "__main__":
    op = MtsBatchOp()

    ep_path = "entity_product"
    ret_eps = op.search_endpoints(ep_path)
    for ep in ret_eps:
        print(f"endpoint={ep["path"]}, id={ep['id']}")

    # ep_config = op.get_endpoint_config(44623)
    # print("endpoint config:", ep_config)

    # op.update_enpoint_by_id(44623, timeout_sec=10)

    # op.copy_endpoint(ep_path, "staging")
