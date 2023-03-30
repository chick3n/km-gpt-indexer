python -m uvicorn api:app --reload


# Change these values to the ones used to create the App Service.
az webapp deploy --name 'dc-assistant' --resource-group 'Demo' --src-path ./src.zip

gunicorn -w 2 -k uvicorn.workers.UvicornWorker api:app