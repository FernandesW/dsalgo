# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time


class Screen:
    new_crap = "Hi"
    links = {
        1: "http://localhost:5000/",
        2: "http://127.0.0.1:5000/",
        3: "http://localhost:5000/",
        4: "http://127.0.0.1:5000/",
        5: "http://localhost:5000/",
        6: "https://github.com/SeleniumHQ/selenium/blob/trunk/py/selenium/common/exceptions.py/",
        7: "https://reactjs.org/",
        8: "http://127.0.0.1:5000/",
        9: "https://github.com/SeleniumHQ/selenium/blob/trunk/py/selenium/common/exceptions.py/",
        10: "https://reactjs.org/",

    }

    def take_logs(self):
        from selenium import webdriver
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import WebDriverException
        from webdriver_manager.chrome import ChromeDriverManager
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        driver.get("http://localhost:3000")
        WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        driver.execute_script("localStorage.setItem('key','value');")
        time.sleep(10)
        driver.close()


    def take_screenshot(self):
        from selenium import webdriver
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import WebDriverException
        from webdriver_manager.chrome import ChromeDriverManager
        import numpy as np
        import cv2
        import pyautogui
        import time
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        errors = []
        for k, v in self.links.items():
            try:
                driver.get(v)
                WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            except WebDriverException:
                print(f'Error connection refused for {v}')
                errors.append(v)
                time.sleep(2)
                continue
            time.sleep(2)
            image = pyautogui.screenshot()
            image = cv2.cvtColor(np.array(image),
                                 cv2.COLOR_RGB2BGR)
            values = driver.execute_script("return document.getElementsByTagName('body')[0].innerText")
            values = values.split("\n")
            if len(values) > 1:
                errors.append(v)
            # print(values)
            # print(len(values))
            #
            kk = v.split('/')[3]
            # print(f'prod_screenshots/{k}_{kk}.png')
            #
            cv2.imwrite(f'prod_screenshots/{k}_{kk}.png', image)
            time.sleep(3)
        print("Final Errors")
        print(errors)
        driver.close()

    def make_text_to_excel(self):
        print(self.new_crap)
        import json
        with open('read.txt', 'r') as f:
            for line in f:
                club = ""
                # strings = ("POST","PUT")
                # if any(indiv in line for indiv in strings):
                if "POST" in line:
                    print("In Post:" + line, end="")
                elif "payload" in line:
                    print("In Payload:" + line, end="")
                    while True:
                        payload_lines = f.readline()
                        club += payload_lines
                        if "}" in payload_lines:
                            break
                    res_dict = json.loads(club)
                    print(res_dict['cbc'])

def check():
    print("Done Printing")


def filter_data():
    pass


if __name__ == '__main__':
    obj = Screen()
    # obj.take_logs()
    # obj.take_screenshot()
    obj.make_text_to_excel()
    print(check())
    print(filter_data())
