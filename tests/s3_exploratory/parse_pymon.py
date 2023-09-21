"""
Parse and display test memory.

Uses pytest-monitor plugin from https://github.com/CFMTech/pytest-monitor
Lots of other metrics can be read from the file via sqlite parsing.,
currently just MEM_USAGE (RES memory, in MB).
"""
import sqlite3
import sys
from operator import itemgetter


def _get_big_mem_tests(cur):
    """Find out which tests are heavy on memory."""
    big_mem_tests = []
    print("Test name / memory (MB)")
    for row in cur.execute('select ITEM, MEM_USAGE from TEST_METRICS;'):
        test_name, memory_used = row[0], row[1]
        # print all tests with mem (RES mem)
        print(test_name, memory_used)


def _get_slow_tests(cur):
    """Find out which tests are slow."""
    timed_tests = []
    sq_command = \
        'select ITEM, ITEM_VARIANT, ITEM_PATH, TOTAL_TIME from TEST_METRICS;'
    for row in cur.execute(sq_command):
        test_name, test_var, test_path, time_used = \
            row[0], row[1], row[2], row[3]
        timed_tests.append((test_name, test_var, test_path, time_used))

    timed_tests = sorted(timed_tests, reverse=True, key=itemgetter(3))
    hundred_slowest_tests = timed_tests[0:100]
    print("List of 100 slowest tests (duration, path, name")
    if hundred_slowest_tests:
        for _, test_var, pth, test_duration in hundred_slowest_tests:
            pth = pth.replace(".", "/") + ".py"
            executable_test = pth + "::" + test_var
            if ", " in executable_test:
                executable_test = executable_test.replace(", ", "-")
            mssg = "%.2f" % test_duration + "s " + executable_test
            print(mssg)
    else:
        print("Could not retrieve test timing data.")


def _parse_pymon_database():
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect("../.pymon")
    cur = con.cursor()

    # The result of a "cursor.execute" can be iterated over by row
    # first look at memory
    _get_big_mem_tests(cur)

    # then look at total time (in seconds)
    # (user time is availbale too via USER_TIME, kernel time via KERNEL_TIME)
    _get_slow_tests(cur)

    # Be sure to close the connection
    con.close()


if __name__ == '__main__':
    _parse_pymon_database()
