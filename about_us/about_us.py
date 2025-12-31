from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, os

# ================= HELPER =================
def clean_name(text):
    return "".join(c for c in text if c.isalnum() or c in (" ", "_", "-")).strip()

def write_table(out, table):
    rows = table.find_elements(By.XPATH, ".//tr")
    table_data = []

    for row in rows:
        cells = row.find_elements(By.XPATH, ".//th | .//td")
        row_data = [cell.text.strip() for cell in cells]
        if row_data:
            table_data.append(row_data)

    if not table_data:
        return

    col_widths = [
        max(len(row[i]) for row in table_data if i < len(row))
        for i in range(len(table_data[0]))
    ]

    for row in table_data:
        line = " | ".join(
            row[i].ljust(col_widths[i]) for i in range(len(row))
        )
        out.write(line + "\n")

    out.write("-" * 100 + "\n")


# ================= SETUP =================
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)

url = "https://www.sunbeaminfo.in/about-us"
print(f"Scraping: {url}")

try:
    driver.get(url)
    time.sleep(3)

    # ================= PAGE HEADING =================
    page_heading = driver.find_element(By.TAG_NAME, "h1").text.strip()
    safe_page_heading = clean_name(page_heading)

    page_folder = os.path.join("data", safe_page_heading)
    os.makedirs(page_folder, exist_ok=True)

    # =================================================
    # 1️⃣ PAGE CONTENT (TEXT + TABLES)
    # =================================================
    page_file = os.path.join(page_folder, f"{safe_page_heading}.txt")

    with open(page_file, "w", encoding="utf-8") as out:
        out.write("=" * 100 + "\n")
        out.write(f"PAGE: {page_heading}\n")
        out.write(f"URL: {url}\n")
        out.write("=" * 100 + "\n\n")

        # ---------- TEXT ----------
        text_elements = driver.find_elements(
            By.XPATH,
            "//div[contains(@class,'container')]//h2 | "
            "//div[contains(@class,'container')]//h3 | "
            "//div[contains(@class,'container')]//p | "
            "//div[contains(@class,'container')]//li"
        )

        seen = set()
        for el in text_elements:
            txt = el.text.strip()
            if txt and txt not in seen:
                seen.add(txt)
                out.write(txt + "\n")

        # ---------- TABLES ----------
        tables = driver.find_elements(
            By.XPATH,
            "//div[contains(@class,'table-responsive')]//table"
        )

        for i, table in enumerate(tables, start=1):
            out.write("\n" + "=" * 100 + "\n")
            out.write(f"TABLE {i}\n")
            out.write("=" * 100 + "\n")
            write_table(out, table)

    print(f"✅ Page content saved: {page_file}")

    # =================================================
    # 2️⃣ ACCORDION / BUTTON CONTENT (TEXT + TABLES)
    # =================================================
    accordion_links = driver.find_elements(
        By.XPATH,
        "//a[contains(@data-toggle,'collapse') and contains(@href,'#collapse')]"
    )

    for acc in accordion_links:
        section_title = acc.text.strip()
        if not section_title:
            continue

        safe_section = clean_name(section_title)
        section_file = os.path.join(page_folder, f"{safe_section}.txt")

        try:
            collapse_id = acc.get_attribute("href").split("#")[-1]

            driver.execute_script("arguments[0].scrollIntoView(true);", acc)
            driver.execute_script("arguments[0].click();", acc)

            panel = wait.until(
                EC.visibility_of_element_located((By.ID, collapse_id))
            )

            time.sleep(1)

            with open(section_file, "w", encoding="utf-8") as out:
                out.write("=" * 100 + "\n")
                out.write(f"SECTION: {section_title}\n")
                out.write(f"PAGE: {page_heading}\n")
                out.write("=" * 100 + "\n\n")

                # ---- TEXT ----
                items = panel.find_elements(By.XPATH, ".//p | .//li")
                for item in items:
                    txt = item.text.strip()
                    if txt:
                        out.write(txt + "\n")

                # ---- TABLES ----
                tables = panel.find_elements(
                    By.XPATH,
                    ".//div[contains(@class,'table-responsive')]//table"
                )

                for i, table in enumerate(tables, start=1):
                    out.write("\n" + "=" * 80 + "\n")
                    out.write(f"TABLE {i}\n")
                    out.write("=" * 80 + "\n")
                    write_table(out, table)

            print(f"✅ Section saved: {section_file}")

        except Exception as e:
            print(f"❌ Section skipped: {section_title} | {e}")

except Exception as e:
    print(f"❌ Failed: {e}")

driver.quit()
print("\n SCRAPING COMPLETED SUCCESSFULLY ")