# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Get the newest commit SHA value from github."""

import logging

import requests

logging.basicConfig(level=logging.INFO)


def fetch_sha_from_url(url):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            return data.get('sha')
        else:
            logging.info(
                f"Failed to fetch data. Status code: {response.status_code}"
            )
    except requests.RequestException as e:
        logging.info(f"Error during request: {e}")

    return None


if __name__ == '__main__':

    url = 'https://api.github.com/repos/PaddlePaddle/Paddle/commits/develop'

    sha_value = fetch_sha_from_url(url)
    if sha_value:
        logging.info(f"The SHA value is:{sha_value}")
        with open('PaddleShaValue.txt', 'w') as file:
            file.write(sha_value)
    else:
        logging.info("Failed to fetch or parse SHA value.")
