import unittest
from unittest.mock import patch

import openai
from tenacity import RetryError

from process_transcript import chat_completions_request


class TestChatCompletionRequest(unittest.TestCase):
    retry_count = 3

    @patch('process_transcript.client.chat.completions.create')
    def test_api_error_request(self, mock_create):
        messages = [{"role": "system", "content": "You are helpful assistant."}]

        mock_create.side_effect = openai.APIConnectionError(request=None)

        with self.assertRaises(RetryError):
            chat_completions_request(messages)

        self.assertEqual(mock_create.call_count, self.retry_count)
