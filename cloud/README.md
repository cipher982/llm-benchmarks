# How to Run
1. Create an `.env` file from `.env.example` and fill in the necessary details. You only need to fill in the specific providers you want. Some providers such as Azure annoyingly give you a unique `base_url` and `api_key` for each model you want to use. You can find these details in the Azure portal.
2. Run `docker compose -f docker-compose.cloud.yml up --build` to start the server. Everything should build and start up correctly. But I have only tested this on my VPS running Ubuntu 22.04. If you run into any issues, please let me know.
3. The FastAPI server will now be acceping request with the env var `FASTAPI_PORT_CLOUD` which I have set at `5004`. You can change this in the `.env` file. The simplest way to build the requests is using the api I put together in `api/llm_bench`. Go into `api/` and create a new Python environment, and install via `poetry install` using the `pyproject.toml` file.
4. Once the environment is installed and activated, you can run benchmarks with `run_cloud.py`. For example to run `openai` you can call `python run_cloud.py --providers openai`. You can optionally use `--providers all` and it will operate multiple runs concurrently based on how many `uvicorn` workers you have set in the `Dockerfile-cloud` file. I have set this to `8` by default.
5. The results will be primarily logged to MongoDB. This isn't my ideal method but the initial stages involved a lot of schema changes and this was simple to manage at the beginning. I will be moving to a more structured database in the future. You can also check the `logs/` folder for a printed out version of the results. This is useful for debugging and checking the results of the runs as they happen.


## Logging
There are a few ways to view progress:
- `run_cloud.py` output: this simply shows a pass or fail for each request with some basic information.
- `./logs/benchmarks_cloud.log`: shows all the metrics and information for each request as a simple text file.
- `./logs/benchmarks_cloud.json`: same as above but formats into a more machine parsable json file.
- MongoDB: if you set the `.env` variable `LOG_TO_MONGO` as True, the logs will be stored in a MongoDB database. If this is the case, you will be required to also provide `MONGODB_URI`, `MONGODB_DB`, and `MONGODB_COLLECTION_CLOUD` in the `.env` file. This is my primary logging system as it enables me to view realtime results in my react frontend I also built for this project at [llm-benchmarks.com](https://llm-benchmarks.com). It also enables the feature where the benchmarking script will check if a particular model config has been run before and skip it unless you set the run param `--run-always`. This is useful for debugging and testing new models.
