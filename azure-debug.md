# Azure Debug Cheat Sheet

Live debugging commands for the `<FUNCTION_APP_NAME>` function app (Flex Consumption), the `<STORAGE_ACCOUNT>` storage account, and the Event Grid blob trigger. Run in Cloud Shell (`az`) unless noted.

Placeholders: `<SUBSCRIPTION_ID>` — your subscription id (`az account show --query id -o tsv`); `<APP_DOMAIN>` — the app's unique domain (portal → function app → Overview → URL); `<PRINCIPAL_ID>` — the app identity's principal id; `<FUNCTION_APP_NAME>`, `<STORAGE_ACCOUNT>`, `<RESOURCE_GROUP>` — your resource names.

## Host errors from Application Insights

Fetches exceptions and error traces (the crash-loop stack trace lives here):

```bash
az monitor app-insights query --app <FUNCTION_APP_NAME> --resource-group <RESOURCE_GROUP> --analytics-query "union (exceptions | project timestamp, severityLevel=4, msg=outerMessage), (traces | where severityLevel >= 3 | project timestamp, severityLevel, msg=message) | where timestamp > ago(2h) | order by timestamp desc | take 40" -o table
```

Portal alternative: function app → **Log stream**, read the lines after `An unhandled exception has occurred. Host is shutting down.`

## Event Grid subscription

```bash
az eventgrid event-subscription list --source-resource-id "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Storage/storageAccounts/<STORAGE_ACCOUNT>" -o table
```

The subscription **must** filter the subject to the uploads container — without it, every write to `results` re-triggers the function (`processed_processed_…` feedback loop). Delete and recreate it with the filter:

```bash
az eventgrid event-subscription delete --name <SUB_NAME> --source-resource-id "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Storage/storageAccounts/<STORAGE_ACCOUNT>"
az eventgrid event-subscription create --name <SUB_NAME> --source-resource-id "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Storage/storageAccounts/<STORAGE_ACCOUNT>" --endpoint-type webhook --endpoint "https://<APP_DOMAIN>/runtime/webhooks/blobs?functionName=Host.Functions.analyze_image&code=<blobs_extension key>" --included-event-types Microsoft.Storage.BlobCreated --subject-begins-with "/blobServices/default/containers/uploads/blobs/"
```

Resource provider status / registration (required before event subscriptions can exist):

```bash
az provider show --namespace Microsoft.EventGrid --query registrationState
az provider register --namespace Microsoft.EventGrid
```

## Function app lifecycle

```bash
az functionapp restart --name <FUNCTION_APP_NAME> --resource-group <RESOURCE_GROUP>
az functionapp stop --name <FUNCTION_APP_NAME> --resource-group <RESOURCE_GROUP> && sleep 30 && az functionapp start --name <FUNCTION_APP_NAME> --resource-group <RESOURCE_GROUP>
```

## Health probes (local terminal, curl)

HTTP functions (http group):

```bash
curl -s -o /dev/null -w "HTTP %{http_code} in %{time_total}s\n" "https://<APP_DOMAIN>/api/hello?name=warmup"
```

Blob trigger webhook (blob group). Key: portal → `<FUNCTION_APP_NAME>` → Functions → App keys → System keys → `blobs_extension`:

```
https://<APP_DOMAIN>/runtime/webhooks/blobs?functionName=Host.Functions.analyze_image&code=<blobs_extension key>
```

## Symptom → meaning

| Observation | Meaning |
| --- | --- |
| `hello` → 503 after exactly ~60s | No ready instance: host crash loop or deployment in progress |
| `hello` → 200 but webhook → 503 | Blob group instance wedged (stop + start the app) |
| Webhook → 500, empty body | Blob extension rejected the delivery (trigger not registered as Event Grid, or startup failure) |
| Blob trigger never invoked, zero invocations | Flex does not run polling-based blob triggers — must be `source="EventGrid"` |
| Old blobs never processed | Blob triggers (Event Grid) only fire on blobs created after the trigger exists |
| 403 `AuthorizationPermissionMismatch` from `Windows-Azure-Queue` in App Insights | App identity lacks the **queue** data role — the Event Grid blob trigger uses the Queue service internally; host crash-loops at startup |
| `results` fills with `processed_processed_…` blobs | event subscription has no subject filter — the function re-triggers on its own `results` writes (feedback loop) | recreate the subscription with `--subject-begins-with "/blobServices/default/containers/uploads/blobs/"`, then wipe `results` |

## Identity roles for identity-based storage (AzureWebJobsStorage__*)

The portal app-storage flow grants only `Storage Blob Data Contributor`. The Event Grid blob trigger additionally needs queue access:

```bash
az functionapp identity show --name <FUNCTION_APP_NAME> --resource-group <RESOURCE_GROUP>
az role assignment create --assignee-object-id <PRINCIPAL_ID> --assignee-principal-type ServicePrincipal --role "Storage Queue Data Contributor" --scope "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Storage/storageAccounts/<STORAGE_ACCOUNT>"
az role assignment create --assignee-object-id <PRINCIPAL_ID> --assignee-principal-type ServicePrincipal --role "Storage Table Data Contributor" --scope "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.Storage/storageAccounts/<STORAGE_ACCOUNT>"
```

## Blob container inspection (local terminal, python)

Requires `AZURE_STORAGE_CONNECTION_STRING` from `deploy.env` and the `azure-storage-blob` package (use the `azure-function` venv):

```python
from azure.storage.blob import BlobServiceClient

conn = [l for l in open("../deploy.env") if l.startswith("AZURE_STORAGE_CONNECTION_STRING=")][0].split("=", 1)[1].strip()
service = BlobServiceClient.from_connection_string(conn)
for blob in service.get_container_client("uploads").list_blobs():
    print(blob.name, blob.size, blob.last_modified)
```

Delete all blobs in a container:

```bash
CONN=$(az storage account show-connection-string --name <STORAGE_ACCOUNT> --resource-group <RESOURCE_GROUP> -o tsv)
az storage blob delete-batch --source uploads --connection-string "$CONN"
az storage blob delete-batch --source results --connection-string "$CONN"
```