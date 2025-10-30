from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from typing import List
import re

# Headers for requests fallback (if needed)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
BASE_URL = "https://www.gsmarena.com"

def start_driver():
    """Starts a Selenium WebDriver instance with anti-detection options."""
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    # Keep headless off for now to see what's happening
    # options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(f'user-agent={HEADERS["User-Agent"]}')

    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        print("Selenium WebDriver started successfully.")
        return driver
    except Exception as e:
        print(f"Error starting Selenium WebDriver: {e}")
        return None

def find_user_reviews_page(driver, phone_name: str) -> str:
    """Uses Selenium to search GSMArena and find the user reviews page URL."""
    search_url = f"https://www.gsmarena.com/res.php3?sSearch={phone_name.replace(' ', '+')}"
    print(f"Selenium searching URL: {search_url}")
    try:
        driver.get(search_url)
        WebDriverWait(driver, 15).until( # Increased wait
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.makers a, #review-body ul a"))
        )

        first_device_link_element = None
        try:
             first_device_link_element = driver.find_element(By.CSS_SELECTOR, "div.makers a")
        except NoSuchElementException:
             try:
                 first_device_link_element = driver.find_element(By.CSS_SELECTOR, "#review-body ul a")
             except NoSuchElementException:
                print("Could not find the first device link in search results.")
                return None

        device_page_url = first_device_link_element.get_attribute("href")
        if not device_page_url:
            print("Found device link element, but it has no href.")
            return None

        # Ensure URL is absolute
        if not device_page_url.startswith(('http://', 'https://')):
            device_page_url = BASE_URL + "/" + device_page_url.lstrip('/')


        print(f"Selenium found device page: {device_page_url}")
        driver.get(device_page_url)

        # Wait for the device page to load and find the "Read all opinions" link
        reviews_link_selector = "a[href*='-reviews-'][href$='.php']"
        WebDriverWait(driver, 15).until( # Increased wait
             EC.element_to_be_clickable((By.CSS_SELECTOR, reviews_link_selector))
        )
        reviews_link_element = driver.find_element(By.CSS_SELECTOR, reviews_link_selector)
        reviews_page_url = reviews_link_element.get_attribute("href")

        if not reviews_page_url:
             print("Found reviews link element, but it has no href.")
             return None

        # Ensure URL is absolute
        if not reviews_page_url.startswith(('http://', 'https://')):
            reviews_page_url = BASE_URL + "/" + reviews_page_url.lstrip('/')

        print(f"Selenium found user reviews page: {reviews_page_url}")
        return reviews_page_url

    except TimeoutException:
        print("Timeout waiting for elements on GSMArena search or device page.")
        # Try capturing screenshot
        try:
             driver.save_screenshot("debug_timeout_search.png")
             print("Screenshot saved as debug_timeout_search.png")
        except:
             pass # Ignore if screenshot fails
        return None
    except NoSuchElementException as e:
        print(f"Could not find necessary elements: {e}")
        # Try capturing screenshot
        try:
            driver.save_screenshot("debug_noelement_search.png")
            print("Screenshot saved as debug_noelement_search.png")
        except:
             pass # Ignore if screenshot fails
        return None
    except Exception as e:
        print(f"An error occurred during Selenium navigation: {e}")
        return None


def scrape_reviews_with_selenium(reviews_page_url: str, driver, max_pages: int = 5) -> List[str]:
    """Scrapes user reviews from the given URL using Selenium."""
    all_reviews = []
    current_url = reviews_page_url

    for page in range(max_pages):
        print(f"Selenium scraping page {page + 1}/{max_pages} from URL: {current_url}")
        try:
            driver.get(current_url)
            
            # [--- FIX #1: Wait specifically for the reviews container ---]
            reviews_container_selector = "div.user-thread" # This div holds the reviews
            WebDriverWait(driver, 15).until( # Increased timeout
                EC.presence_of_element_located((By.CSS_SELECTOR, reviews_container_selector))
            )
            print("Reviews container found.")
            
            # Optional: Scroll down a bit to ensure content loads
            driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(1) # Small pause after scroll

            # Get page source AFTER waiting and scrolling
            page_html = driver.page_source
            soup = BeautifulSoup(page_html, 'lxml')

            # [--- FIX #2: Refined Selector for review text ---]
            # Find all review containers first
            review_containers = soup.select('div.user-thread')
            
            if not review_containers:
                print(f"Page {page + 1}: No review containers ('div.user-thread') found even after waiting.")
            else:
                print(f"Page {page + 1}: Found {len(review_containers)} review containers.")
                for container in review_containers:
                    # Find the paragraph directly inside the user-thread div that contains the text
                    # Often the structure is div.user-thread > p
                    review_p = container.find('p', recursive=False) # recursive=False looks only at direct children
                    if review_p:
                        review_text = review_p.text.strip()
                        if review_text:
                            all_reviews.append(review_text)
                    else:
                         # Sometimes the text might be directly in the div or another tag
                         review_text_alt = container.get_text(separator=' ', strip=True)
                         # Check if it's not just metadata like username/date
                         if review_text_alt and len(review_text_alt) > 50: # Arbitrary length check
                            all_reviews.append(review_text_alt)
                         else:
                            print("Could not find review text paragraph inside a container.")

            # [--- FIX #3: More robust pagination finding ---]
            try:
                # Find the 'Next page' link using selenium *before* checking soup object
                next_page_element = driver.find_element(By.CSS_SELECTOR, 'a.pages-next')
                next_page_url = next_page_element.get_attribute('href')
                if not next_page_url:
                    print("Next page link found but has no href. Scraping finished.")
                    break

                current_url = next_page_url # Update URL for the next loop iteration
                print("Found next page link via Selenium.")
                # No need to click, just get the URL and loop

            except NoSuchElementException:
                print("Next page link not found using Selenium. Scraping finished.")
                break
            # [--- FIX #3 ENDS HERE ---]

        except TimeoutException:
            print(f"Timeout waiting for elements on reviews page: {current_url}")
            try:
                 driver.save_screenshot(f"debug_timeout_page_{page+1}.png")
                 print(f"Screenshot saved as debug_timeout_page_{page+1}.png")
            except:
                 pass
            break # Stop if a page times out
        except Exception as e:
            print(f"An unexpected error occurred during Selenium scraping page {page + 1}: {e}")
            break

    return all_reviews


# Main function callable by FastAPI (no change here)
def scrape_gsmarena_reviews_selenium(phone_name: str, max_pages: int = 5) -> List[str]:
    driver = start_driver()
    if not driver:
        return ["Error: Could not start Selenium WebDriver."]

    reviews_page_url = find_user_reviews_page(driver, phone_name)
    reviews = []
    if reviews_page_url:
        reviews = scrape_reviews_with_selenium(reviews_page_url, driver, max_pages)
    else:
        # If find_user_reviews_page returned None, provide a more specific error
        reviews = [f"Error: Could not find the user reviews page URL for '{phone_name}'."]

    print(f"Closing Selenium WebDriver. Found {len(reviews)} reviews for '{phone_name}'.")
    driver.quit()
    # Check again if reviews list contains only an error message
    if reviews and reviews[0].startswith("Error:"):
        return reviews # Pass the specific error message
    elif not reviews:
        return ["Error: No reviews were successfully scraped."] # Generic message if list is empty
    else:
        return reviews


# --- Test karne ke liye ---
if __name__ == "__main__":
    test_phone = "Samsung Galaxy S24 Ultra"
    print(f"\n--- Testing Selenium Scraper for: {test_phone} ---")
    reviews_result = scrape_gsmarena_reviews_selenium(test_phone, max_pages=1)
    print("\n--- Scraped Reviews Sample ---")
    if reviews_result and not reviews_result[0].startswith("Error:"):
        for i, review in enumerate(reviews_result[:5]):
            print(f"{i+1}: {review[:150]}...")
    else:
        print(f"Test failed or no reviews found: {reviews_result}")