# -- coding: utf-8 --

import json
import os
import time
from typing import Any, Final, Optional, TypedDict

import requests
from dotenv import load_dotenv
from requests.exceptions import HTTPError
from requests.models import Response

load_dotenv()


class ServiceInfo(TypedDict):
    label: str
    value: int
    key: int


class EndpointInfo(TypedDict):
    protocol: str
    productLine: str
    service_info: ServiceInfo


class Endpoint(TypedDict):
    info_object: EndpointInfo
    path: str
    method: str
    domain: str
    selected_approvers: list[str]
    change_comment: str
    endpoint: str
    is_formily: bool


class EndpointItem(TypedDict):
    id: int
    path: str


class EndpointConfig(TypedDict):
    endpoint_id: int
    environments: list[str]
    gateway_config: str
    selected_approvers: list[str]
    change_comment: str
    is_formily: bool


class GatewayConfig(TypedDict):
    type: str
    spex: str
    raw_request: bool
    timeout_sec: int
    serve_rule: dict
    version: int
    auth_type: str
    login_required: bool
    permissions: list[Any]
    permissions_logical_connective: str
    seller_cid_routing_config: dict
    cors_header_configs: dict
    need_meta: list[str]
    need_cookies: list[str]
    param_type: dict
    need_headers: list[str]
    cookie_control: str


class MtsBatchOp:

    def __init__(self, env: str, approver: str) -> None:
        if len(env) == 0 or len(approver) == 0:
            raise ValueError("parameter env or approver is null")

        self.pop_domain: Final[str] = os.getenv("POP_DOMAIN", "")
        self.sp_host: Final[str] = os.getenv("SP_HOST", "")
        self.sp_token: Final[str] = os.getenv("SP_TOKEN", "")
        if not all((self.pop_domain, self.sp_host, self.sp_token)):
            raise ValueError("pop_domain, sp_host, sp_token is empty")

        self.env: Final[str] = env
        self.approver: Final[str] = approver

        self.session = requests.Session()
        self.session.headers.update(
            {
                "authorization": f"Bearer {self.sp_token}",
                "content-type": "application/json",
            }
        )

    def get_first_endpoint(self, keyword: str) -> Optional[EndpointItem]:
        endpoints = self.search_endpoints(keyword)
        return endpoints[0] if len(endpoints) > 0 else None

    def search_endpoints(self, keyword: str) -> list[EndpointItem]:
        if not keyword:
            return []

        url: str = self.sp_host + "/v1/ecp_nonlive/get_endpoints_v2"
        data: dict = {
            "domain": self.pop_domain,
            "need_config_statuses": True,
            "path": keyword,
        }
        resp: Response = self.session.post(url, json=data)
        resp_data = self.check_and_get_resp_data(resp)

        endpoints: list[EndpointItem] = []
        for ep in resp_data["endpoints"]:
            endpoints.append(EndpointItem(id=ep["id"], path=ep["endpoint"]))
        return endpoints

    def get_endpoint_gateway_config(self, endpoint_id: int) -> dict:
        if endpoint_id <= 0:
            raise ValueError(f"invalid endpoint_id: {endpoint_id}")

        url = f"{self.sp_host}/v1/ecp_nonlive/endpoint_config/get?endpoint_id={endpoint_id}&environment={self.env}&is_formily=false"
        resp: Response = self.session.get(url)
        resp_data = self.check_and_get_resp_data(resp)

        gateway_config = resp_data["gateway_config"]
        # print(f"[debug] get gateway config for endpoint [{endpoint_id}]: {gateway_config}")
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
        gateway_config = self.get_endpoint_gateway_config(endpoint_id)

        # do update
        if kwargs:
            for k, v in kwargs.items():
                print(f"update gateway config [{k}]: {gateway_config[k]}->{v}")
                gateway_config[k] = v

        data = EndpointConfig(
            endpoint_id=endpoint_id,
            environments=[self.env],
            gateway_config=json.dumps(gateway_config),
            selected_approvers=[self.approver],
            change_comment="update gateway config",
            is_formily=False,
        )

        # print("[debug] update endpoint request:", data)
        url = self.sp_host + "/v1/ecp_nonlive/endpoint_config/update_multi_envs"
        resp: Response = self.session.post(url, json=data)
        resp_data = self.check_and_get_resp_data(resp)
        print(
            f"update endpoint success: endpoint_id={endpoint_id}, resp_body={resp_data}"
        )

    def copy_endpoint(
        self, endpoint_path: str, to_env: str, change_comment: str
    ) -> None:
        if not to_env:
            raise ValueError("copy endpoint to_env is null")

        ep = self.get_first_endpoint(endpoint_path)
        if not ep or not ep.get("id"):
            raise KeyError(f"no endpoint found by path: {endpoint_path}")

        gateway_config = self.get_endpoint_gateway_config(ep.get("id"))
        data = EndpointConfig(
            endpoint_id=ep.get("id"),
            environments=[to_env],
            gateway_config=json.dumps(gateway_config),
            selected_approvers=[self.approver],
            change_comment=change_comment,
            is_formily=False,
        )

        url = self.sp_host + "/v1/ecp_nonlive/endpoint_config/update_multi_envs"
        resp: Response = self.session.post(url, json=data)
        resp_data = self.check_and_get_resp_data(resp)
        print(
            f"copy endpoint to new env success: endpoint_path={endpoint_path}, to_env={to_env} resp_body={resp_data}"
        )

    def create_endpoint(self, ep_path: str, change_comment: str) -> None:
        if len(ep_path) == 0 or len(change_comment) == 0:
            raise ValueError("parameter path or change_comment is null")

        url = self.sp_host + "/v1/ecp_nonlive/endpoint_info/create"
        data = self._build_endpoint_request(ep_path, ep_path, change_comment)
        resp: Response = self.session.post(url, json=data)
        resp_data = self.check_and_get_resp_data(resp)
        print(f"create endpoint success: path={ep_path}, resp_body={resp_data}")

    def _build_endpoint_request(
        self, path: str, endpoint: str, change_comment: str
    ) -> Endpoint:
        info_object = EndpointInfo(
            protocol="HTTP",
            productLine="Promotion",
            service_info=ServiceInfo(
                label=os.getenv("PJ_PORTAL_ID", ""),
                value=15672,
                key=15672,
            ),
        )
        return Endpoint(
            info_object=info_object,
            path=path,
            method="POST",
            domain=self.pop_domain,
            selected_approvers=[self.approver],
            change_comment=change_comment,
            endpoint=endpoint,
            is_formily=True,
        )

    def create_endpoint_config(
        self, ep_path: str, spex_cmd: str, change_comment: str
    ) -> None:
        ep = self.get_first_endpoint(ep_path)
        if not ep or not ep.get("id"):
            raise ValueError(f"endpoint is not found: {ep_path}")

        url = self.sp_host + "/v1/ecp_nonlive/endpoint_config/create_multi_envs"
        data = self._build_endpoint_config_request(
            ep.get("id"), spex_cmd, change_comment
        )
        resp: Response = self.session.post(url, json=data)
        resp_data = self.check_and_get_resp_data(resp)
        print(
            f"create gateway config success: path={ep_path}, endpoint_id={ep.get("id")}, resp_body={resp_data}"
        )

    def _build_endpoint_config_request(
        self, endpoint_id: int, spex_cmd: str, change_comment: str
    ) -> EndpointConfig:
        gateway_config = GatewayConfig(
            type="spex",
            spex=spex_cmd,
            raw_request=False,
            timeout_sec=30,
            serve_rule={"with_cid": True, "with_idc": True},
            version=2,
            auth_type="soup",
            login_required=True,
            permissions=[],
            permissions_logical_connective="AND",
            seller_cid_routing_config={},
            cors_header_configs={},
            need_meta=["soup_email"],
            need_cookies=[],
            param_type={},
            need_headers=[],
            cookie_control="disabled",
        )

        return EndpointConfig(
            endpoint_id=endpoint_id,
            environments=[self.env],
            gateway_config=json.dumps(gateway_config),
            selected_approvers=[self.approver],
            change_comment=change_comment,
            is_formily=False,
        )

    def check_and_get_resp_data(self, resp: Response) -> dict:
        resp.raise_for_status()
        resp_json: dict = resp.json()
        if resp_json["error"]:
            raise HTTPError(
                f"request to mts failed: error_code={resp_json["error"]}, error_msg={resp_json["error_msg"]}"
            )
        return resp_json["data"]


# Main


def get_approver() -> str:
    email = os.getenv("POP_APPROVER")
    if not email:
        raise ValueError("approver is not set in env")
    return email


def main_test():
    op = MtsBatchOp("test", get_approver())

    ep_path = "entity_product"
    ret_eps = op.search_endpoints(ep_path)
    for ep in ret_eps:
        print(f"endpoint={ep["path"]}, id={ep['id']}")

    # ep_config = op.get_endpoint_config(44623)
    # print("endpoint config:", ep_config)

    # op.update_enpoint_by_id(44623, timeout_sec=10)

    # op.copy_endpoint(ep_path, "staging")


def main_validate_endpoints():
    op = MtsBatchOp(env="uat", approver=get_approver())

    for ep_path in ["key_project_field/list_options", "key_project/search"]:
        try:
            ep = op.get_first_endpoint(ep_path)
            if ep:
                gateway_cfg = op.get_endpoint_gateway_config(ep["id"])
                cfg = GatewayConfig(**gateway_cfg)
                print(
                    f"\nendpoint [{ep_path}] config: spex={cfg['spex']}, timeout={cfg['timeout_sec']}, auth={cfg['auth_type']}"
                )
                if not cfg["auth_type"]:
                    print(f"endpoint auth_type [{ep_path}] is null")
            time.sleep(0.5)
        except HTTPError as e:
            print(f"validate endpoint [{ep_path}] error: {e}")


def main_update_endpoints():
    op = MtsBatchOp(env="test", approver=get_approver())
    for ep_path in ["key_project_field/list_options", "key_project/search"]:
        try:
            op.update_enpoint(ep_path)
            time.sleep(0.5)
        except HTTPError as e:
            print(f"update endpoint [{ep_path}] error: {e}")


def main_copy_endpoints():
    op = MtsBatchOp(env="test", approver=get_approver())
    for ep_path in ["key_project_field/list_options", "key_project/search"]:
        try:
            op.copy_endpoint(ep_path, to_env="staging", change_comment="add new api")
            time.sleep(0.5)
        except HTTPError as e:
            print(f"copy endpoint [{ep_path}] error: {e}")


def main_create_endpoints():
    op = MtsBatchOp(env="test", approver=get_approver())
    for path, cmd in [
        (
            "/api/gateway/v1/pm_prd_report/get_user_profile",
            "marketplace.pop.roi.service.pm_prd_user_profile_get",
        ),
    ]:
        try:
            op.create_endpoint(path, "add pm.prd.delivery api")
            op.create_endpoint_config(path, cmd, "add pm.prd.delivery api")
            time.sleep(0.5)
        except HTTPError as e:
            print(f"create endpoint [{path}] error: {e}")


if __name__ == "__main__":
    # main_test()

    # main_validate_endpoints()
    # main_update_endpoints()
    # main_create_endpoints()
    pass
