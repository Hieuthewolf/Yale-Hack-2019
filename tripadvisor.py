import random
from selenium import webdriver
from selenium.webdriver.common.proxy import *
from selenium.common.exceptions import ElementClickInterceptedException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait # available since 2.4.0
from selenium.webdriver.support import expected_conditions as EC # available since 2.26.0
from selenium.common.exceptions import WebDriverException, TimeoutException
from twisted.internet.error import ConnectionRefusedError
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import time
import logging

logSplash = logging.getLogger("splash")

class splash(object):
    def __init__(self):
        self.gecko = "C:/Users/Umarbek Nasimov/Desktop/Code_Stuff/geckodriver.exe"
        self.url = "https://www.tripadvisor.com/Airline_Review-d8729099-Reviews-or25-JetBlue#REVIEWS"
        self.proxy_list = 'C:/Users/Umarbek Nasimov/Desktop/UROP/HTraffic/Scrapy_Framework/ProxyFinder/ProxyFinder/proxies.txt'
        if self.proxy_list is None: 
            raise KeyError('PROXY_LIST setting is missing')
        self.proxies = []                                   #proxy list
        fin = open(self.proxy_list)
        try:
            self.proxies = fin.readlines()[0].split(",")[:-1]
        finally:
            fin.close()
        print("===" + str(self.proxies) + "===")
        self.chosen_proxy = random.choice(self.proxies)
        print("Setting splash proxy to " + self.chosen_proxy[8:])
        try:
            self.driver = None
            self.driver = self.new_driver(self.chosen_proxy)
            self.driver.set_page_load_timeout(30)
            self.driver.get("https://httpbin.org/ip")
        except:
            print("EXCEPTION")
            self.change_driver(remove=True)
        self.firebase()
        self.get_info(self.url)
    def firebase(self):
        cred = credentials.Certificate(('C:/Users/Umarbek Nasimov/Desktop/yhack.json'))
        firebase_admin.initialize_app(cred, {'databaseURL' : 'https://yhack-cb990.firebaseio.com/'})

        self.root = db.reference()
        # Add a new user under /users.
        #root.child('reviews').push(review)
    def new_driver(self,proxy):
        print("Changing splash proxy to " + proxy[8:])
        firefox_profile = webdriver.FirefoxProfile()
        firefox_profile.set_preference('permissions.default.image', 2) #ELIMINATE IMAGES
        firefox_profile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
        myProxy = proxy[8:]
        proxy = Proxy({'proxyType': ProxyType.MANUAL,'sslProxy': myProxy,})
        caps = webdriver.DesiredCapabilities.FIREFOX
        proxy.add_to_capabilities(caps)
        return webdriver.Firefox(desired_capabilities=caps,firefox_profile = firefox_profile,executable_path=self.gecko)
    def change_driver(self,remove = False):
        new_proxy = random.choice(self.proxies)
        if remove == True:
            try: 
                self.proxies.remove(self.chosen_proxy)
                print("REMOVING SPLASH PROXY " +self.chosen_proxy)
            except:
                print("FAILED TO REMOVE ")
        while new_proxy == self.chosen_proxy:
            new_proxy = random.choice(self.proxies)
        self.chosen_proxy = new_proxy
        if self.driver:
            self.driver.close()
        self.failed = 0
        try:
            self.driver = self.new_driver(self.chosen_proxy)
            self.driver.set_page_load_timeout(30)
            self.driver.get("https://httpbin.org/ip")
        except:
            self.change_driver(remove=True)
    def get_info(self,url):
        while True:
            try:
                self.driver.get(url)
                break
            except:
                self.change_driver(remove=True)
        time.sleep(0.5)
        print("GETTING INFO")
        review = {} #title, content, stars, domestic or international, to and from cities
        test = WebDriverWait(self.driver,30).until(EC.presence_of_element_located((By.XPATH,'//div[contains(@class,"location-review-review-list-parts-SingleReview__reviewContainer")]')))
        while True:
            for x in self.driver.find_elements_by_xpath('//div[contains(@class,"location-review-review-list-parts-SingleReview__reviewContainer")]'): #each is x
                title = x.find_element_by_xpath('.//a[contains(@class,"location-review-review-list-parts-ReviewTitle__reviewTitleText")]').text
                rating = x.find_element_by_xpath('.//div[contains(@class,"location-review-review-list-parts-RatingLine__bubbles")]//span')
                rating = rating.get_attribute("class") #'ui_bubble_rating bubble_40'
                ratings = {"10","20","30","40","50"}
                for r in ratings:
                    if r in rating:
                        rating = int(r)/10
                        break
                flight = x.find_elements_by_xpath('.//div[contains(@class,"location-review-review-list-parts-RatingLine__labelsContainer")]//div')
                for f in range(len(flight)): #West Palm Beach - Boston, Domestic, Economy
                    t = flight[f].text
                    if f == 0:
                        t = t.split(" - ")
                        departure = t[0]
                        arrival = t[1]
                    if f == 1:
                        domint = t
                    if f == 2:
                        ftype = t   
                try:
                    readmore = x.find_element_by_xpath('.//div[contains(@class,"location-review-review-list-parts-ExpandableReview__containerStyles")]//span[contains(@class,"common-text-ReadMore__cta")]')
                    readmore.click()
                    time.sleep(1)
                except:
                    print("read error")
                    pass

                text = x.find_element_by_xpath('.//div[contains(@class,"location-review-review-list-parts-ExpandableReview__containerStyles")]//q[contains(@class,"location-review-review-list-parts-ExpandableReview__reviewText")]//span').text
                try:
                    date = x.find_element_by_xpath('.//div[contains(@class,"location-review-review-list-parts-EventDate__event_date")]').text
                    #print(date)
                    date = date[date.index(":")+2:]
                except NoSuchElementException:
                    date = None
                try:
                    stats = x.find_elements_by_xpath('.//span[contains(@class,"social-member-MemberHeaderStats__stat")]')[-1].text
                    if "helpful" in stats:
                        stats = stats[:stats.index(" ")]
                    else:
                        raise NoSuchElementException
                except NoSuchElementException:
                    stats = None
                review = {"title":title,"rating":rating,"departure":departure,"arrival":arrival,"domint":domint,"ftype":ftype,"text":text,"date":date,"helpful":stats}
                print(review)
                self.root.child('reviews').push(review)
            try:
                nextpage = self.driver.find_element_by_xpath('//a[contains(@class,"ui_button nav next primary ")]')
                nextpage.click()
                time.sleep(1)
                print("--------------------GOING TO NEXT PAGE----------------------------")
            except NoSuchElementException:
                break
trip = splash()
