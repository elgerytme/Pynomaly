### API Routes Inventory

| Endpoint                   | HTTP Method | Path                 | Request Model       | Response Model        | Dependencies          | Security      |
|----------------------------|-------------|----------------------|---------------------|-----------------------|-----------------------|---------------|
| `/api/v1/health/`          | GET         | /                    | None                | HealthResponse        | get_container         | None          |
| `/api/v1/health/metrics`   | GET         | /metrics             | None                | SystemMetricsResponse | None                  | None          |
| `/api/v1/health/history`   | GET         | /history             | None                | list[dict]            | None                  | None          |
| `/api/v1/health/summary`   | GET         | /summary             | None                | dict                  | None                  | None          |
| `/api/v1/health/ready`     | GET         | /ready               | None                | dict                  | get_container         | None          |
| `/api/v1/health/live`      | GET         | /live                | None                | dict                  | None                  | None          |
| `/api/v1/auth/login`       | POST        | /login               | OAuth2PasswordRequestForm | TokenResponse | get_auth              | None          |
| `/api/v1/auth/refresh`     | POST        | /refresh             | string               | TokenResponse         | get_auth              | None          |
| `/api/v1/auth/register`    | POST        | /register            | RegisterRequest     | UserResponse          | get_auth              | None          |
| `/api/v1/auth/me`          | GET         | /me                  | None                | UserResponse          | require_auth          | Authentication |
| `/api/v1/auth/api-keys`    | POST        | /api-keys            | APIKeyRequest       | APIKeyResponse        | require_auth          | Authentication |
| `/api/v1/auth/api-keys/{api_key}` | DELETE | /api-keys/{api_key} | str                 | dict                  | get_current_user      | Authentication |
| `/api/v1/auth/logout`      | POST        | /logout              | None                | dict                  | get_current_user      | Authentication |
| `/api/v1/datasets/`        | GET         | /                    | None                | list[DatasetDTO]      | require_viewer        | Authorization |
| `/api/v1/datasets/{dataset_id}` | GET   | /{dataset_id}        | UUID                | DatasetDTO            | require_viewer        | Authorization |
| `/api/v1/datasets/upload`  | POST        | /upload              | File, str, str      | DatasetDTO            | require_data_scientist | Authorization |
| `/api/v1/datasets/{dataset_id}/quality` | GET | /{dataset_id}/quality | UUID       | DataQualityReportDTO | require_viewer        | Authorization |
| `/api/v1/datasets/{dataset_id}/sample`  | GET | /{dataset_id}/sample | UUID, int | dict | require_permissions | Authorization |
| `/api/v1/datasets/{dataset_id}/split` | POST | /{dataset_id}/split | UUID, float, int | dict | require_permissions | Authorization |
| `/api/v1/datasets/{dataset_id}` | DELETE | /{dataset_id} | UUID | dict | require_permissions | Authorization |

### Observations

- Circular imports have been noted and mitigated through lazy imports.
- No unusual Pydantic constructs were identified in the inspected files.
