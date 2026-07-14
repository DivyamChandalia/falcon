import unittest
from unittest.mock import patch

from jet.process_args import ProcessArguments


class HeadlessIdentityTests(unittest.TestCase):
    def test_groups_do_not_require_a_tty_or_passwd_entry(self):
        with patch("os.getgroups", return_value=[10122, 20004]):
            self.assertEqual(ProcessArguments._get_user_groups(None), [10122, 20004])


if __name__ == "__main__":
    unittest.main()
