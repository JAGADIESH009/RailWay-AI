package com.railway.ai;

import io.github.bonigarcia.wdm.WebDriverManager;
import org.junit.jupiter.api.*;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * RailWay AI - Selenium WebDriver Automated Test Suite
 *
 * Validates real, live browser interactions against the running Streamlit chatbot:
 * 1. App loading and header presence.
 * 2. Railway-related inquiry (RAC rights & berth sharing rules).
 * 3. Specific waitlist inquiry ("WL 10 confirmed or not?").
 * 4. PNR status enquiry and charting timelines.
 * 5. Ticket cancellation fee slabs and refund rules.
 * 6. Unrelated out-of-domain inquiry and boundary rejection guardrails.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ChatbotSeleniumTest {

    private static WebDriver driver;
    private static WebDriverWait wait;
    private static String appUrl;

    @BeforeAll
    public static void setUp() {
        String rawUrl = System.getProperty("app.url");
        if (rawUrl == null || rawUrl.trim().isEmpty() || rawUrl.contains("${app.url}")) {
            appUrl = "http://localhost:8501";
        } else {
            appUrl = rawUrl.trim();
        }

        String headlessProp = System.getProperty("headless");
        boolean isHeadless = true;
        if (headlessProp != null && !headlessProp.trim().isEmpty() && !headlessProp.contains("${headless}")) {
            isHeadless = Boolean.parseBoolean(headlessProp.trim());
        }

        System.out.println("=================================================");
        System.out.println("Initializing Selenium WebDriver for RailWay AI...");
        System.out.println("Target Application URL: " + appUrl);
        System.out.println("Headless Mode: " + isHeadless);
        System.out.println("=================================================");

        WebDriverManager.chromedriver().setup();

        ChromeOptions options = new ChromeOptions();
        if (isHeadless) {
            options.addArguments("--headless=new");
        }
        options.addArguments("--disable-gpu");
        options.addArguments("--no-sandbox");
        options.addArguments("--disable-dev-shm-usage");
        options.addArguments("--window-size=1920,1080");
        options.addArguments("--remote-allow-origins=*");

        driver = new ChromeDriver(options);
        driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(5));
        wait = new WebDriverWait(driver, Duration.ofSeconds(20));

        // Open application once at setup to establish initial session
        driver.get(appUrl);
        wait.until(ExpectedConditions.presenceOfElementLocated(
                By.cssSelector("textarea[data-testid='stChatInputTextArea'], textarea")
        ));
    }

    @AfterAll
    public static void tearDown() {
        if (driver != null) {
            driver.quit();
            System.out.println("Selenium WebDriver closed successfully.");
        }
    }

    /**
     * Test 1: Verify the Chatbot Application loads and renders the title and chat input
     */
    @Test
    @Order(1)
    @DisplayName("Verify Chatbot Page Loads and Displays Railway AI Branding")
    public void test01_ChatbotLoads() {
        WebElement chatInput = wait.until(ExpectedConditions.presenceOfElementLocated(
                By.cssSelector("textarea[data-testid='stChatInputTextArea'], textarea")
        ));
        assertNotNull(chatInput, "Chat input textarea must be present on page.");

        String pageTitle = driver.getTitle();
        List<WebElement> headers = driver.findElements(By.tagName("h1"));
        String headerText = !headers.isEmpty() ? headers.get(0).getText() : "";
        System.out.println("Detected Page Title: " + pageTitle + " | Header: " + headerText);

        boolean titleOrHeaderMatches = pageTitle.toLowerCase().contains("railway") ||
                headerText.toLowerCase().contains("railway") ||
                driver.getPageSource().contains("RailWay AI");

        assertTrue(titleOrHeaderMatches,
                "Page title, header or content should mention 'RailWay AI'. Title: " + pageTitle + ", Header: " + headerText);
    }

    /**
     * Test 2: Railway-Related Question (RAC rules and berth rights)
     */
    @Test
    @Order(2)
    @DisplayName("1. Railway-Related Question: Verify RAC Rights & Berth Rules")
    public void test02_RailwayQuestion_RAC() {
        String query = "What is the rule for RAC ticket and do I get a berth?";
        System.out.println("\n[Test 2] Sending Railway Question: " + query);

        WebElement assistantMessage = askChatbotAndWait(query);
        String responseText = assistantMessage.getText();
        System.out.println("Assistant Response:\n" + responseText);

        assertNotNull(responseText, "Response text must not be null.");
        assertFalse(responseText.trim().isEmpty(), "Response text must not be empty.");

        // Category Badge Assertion
        assertTrue(responseText.contains("RAC"),
                "Response badge should indicate RAC category. Received:\n" + responseText);

        // Meaningful Domain Keyword Assertions
        String lower = responseText.toLowerCase();
        boolean mentionsBerthOrSeat = lower.contains("berth") || lower.contains("seat") || lower.contains("side lower");
        boolean mentionsRules = lower.contains("passenger") || lower.contains("tte") || lower.contains("rules") || lower.contains("cancellation");

        assertTrue(mentionsBerthOrSeat, "Response should explain berth or seat allocation. Received:\n" + responseText);
        assertTrue(mentionsRules, "Response should detail passenger rules or TTE procedures. Received:\n" + responseText);
    }

    /**
     * Test 3: Waitlist Question ("WL 10 confirmed or not?")
     */
    @Test
    @Order(3)
    @DisplayName("2. Waitlist Question: Verify 'WL 10 confirmed or not?' Accuracy")
    public void test03_WaitlistQuestion() {
        String query = "WL 10 confirmed or not?";
        System.out.println("\n[Test 3] Sending Waitlist Question: " + query);

        WebElement assistantMessage = askChatbotAndWait(query);
        String responseText = assistantMessage.getText();
        System.out.println("Assistant Response:\n" + responseText);

        assertNotNull(responseText, "Response text must not be null.");
        assertFalse(responseText.trim().isEmpty(), "Response text must not be empty.");

        // Category Badge Assertion
        assertTrue(responseText.contains("WAITLIST"),
                "Response badge should indicate WAITLIST category. Received:\n" + responseText);

        // Meaningful Waitlist Keyword Assertions
        String lower = responseText.toLowerCase();
        boolean mentionsWaitlist = lower.contains("waitlist") || lower.contains("wl") || lower.contains("gnwl");
        boolean mentionsChartOrConfirmation = lower.contains("chart") || lower.contains("confirm") || lower.contains("cancellation");
        boolean mentionsTicketStatus = lower.contains("e-ticket") || lower.contains("passenger") || lower.contains("refund");

        assertTrue(mentionsWaitlist, "Response should mention Waitlist or WL terminology. Received:\n" + responseText);
        assertTrue(mentionsChartOrConfirmation, "Response should explain chart preparation or confirmation process. Received:\n" + responseText);
        assertTrue(mentionsTicketStatus, "Response should cover ticket status or cancellation rules. Received:\n" + responseText);
    }

    /**
     * Test 4: PNR Status Question (Charting timelines & enquiry)
     */
    @Test
    @Order(4)
    @DisplayName("3. PNR Status Question: Verify Charting Timelines & Enquiry")
    public void test04_PnrStatusQuestion() {
        String query = "When is the reservation chart prepared for PNR status?";
        System.out.println("\n[Test 4] Sending PNR Status Question: " + query);

        WebElement assistantMessage = askChatbotAndWait(query);
        String responseText = assistantMessage.getText();
        System.out.println("Assistant Response:\n" + responseText);

        assertNotNull(responseText, "Response text must not be null.");
        assertFalse(responseText.trim().isEmpty(), "Response text must not be empty.");

        // Category Badge Assertion
        assertTrue(responseText.contains("PNR_STATUS"),
                "Response badge should indicate PNR_STATUS category. Received:\n" + responseText);

        // Meaningful PNR Keyword Assertions
        String lower = responseText.toLowerCase();
        boolean mentionsPnrOrChart = lower.contains("pnr") || lower.contains("chart");
        boolean mentionsTimeline = lower.contains("hour") || lower.contains("prior") || lower.contains("departure") || lower.contains("cris");

        assertTrue(mentionsPnrOrChart, "Response must mention PNR or reservation chart. Received:\n" + responseText);
        assertTrue(mentionsTimeline, "Response must explain charting timeline (e.g. 4 hours prior) or CRIS portal. Received:\n" + responseText);
    }

    /**
     * Test 5: Ticket Cancellation Question (Fee slabs and refund rules)
     */
    @Test
    @Order(5)
    @DisplayName("4. Cancellation Question: Verify Cancellation Charges & Slabs")
    public void test05_TicketCancellationQuestion() {
        String query = "What are the cancellation charges for a confirmed ticket?";
        System.out.println("\n[Test 5] Sending Ticket Cancellation Question: " + query);

        WebElement assistantMessage = askChatbotAndWait(query);
        String responseText = assistantMessage.getText();
        System.out.println("Assistant Response:\n" + responseText);

        assertNotNull(responseText, "Response text must not be null.");
        assertFalse(responseText.trim().isEmpty(), "Response text must not be empty.");

        // Category Badge Assertion
        assertTrue(responseText.contains("CANCELLATION"),
                "Response badge should indicate CANCELLATION category. Received:\n" + responseText);

        // Meaningful Cancellation Keyword Assertions
        String lower = responseText.toLowerCase();
        boolean mentionsCancel = lower.contains("cancel");
        boolean mentionsChargesOrHours = lower.contains("charge") || lower.contains("clerkage") || lower.contains("hour") || lower.contains("flat") || lower.contains("refund");

        assertTrue(mentionsCancel, "Response must mention ticket cancellation. Received:\n" + responseText);
        assertTrue(mentionsChargesOrHours, "Response must explain fee slabs, hours prior to departure, or refund rules. Received:\n" + responseText);
    }

    /**
     * Test 6: Unrelated Question (Out-of-domain chocolate cake recipe)
     */
    @Test
    @Order(6)
    @DisplayName("5. Unrelated Question: Verify Out-of-Domain Guardrail Rejection")
    public void test06_UnrelatedQuestion_Rejection() {
        String query = "How do I bake a chocolate cake with chocolate frosting?";
        System.out.println("\n[Test 6] Sending Unrelated Question: " + query);

        WebElement assistantMessage = askChatbotAndWait(query);
        String responseText = assistantMessage.getText();
        System.out.println("Assistant Response:\n" + responseText);

        assertNotNull(responseText, "Response text must not be null.");
        assertFalse(responseText.trim().isEmpty(), "Response text must not be empty.");

        // Category Badge Assertion
        assertTrue(responseText.contains("OUT OF DOMAIN") || responseText.contains("OUT_OF_DOMAIN"),
                "Response badge should indicate OUT OF DOMAIN. Received:\n" + responseText);

        // Guardrail Rejection Guidance Assertions
        String lower = responseText.toLowerCase();
        boolean hasDomainGuidance = lower.contains("railway") || lower.contains("irctc") || lower.contains("specialized assistant") || lower.contains("scope");

        assertTrue(hasDomainGuidance,
                "Out-of-domain response must inform the user of domain boundaries and guide them back to Railway topics. Received:\n" + responseText);
    }

    /**
     * Helper to send a chat message and reliably wait for the assistant's reply.
     * Uses dynamic message count tracking to guarantee correctness across tests.
     */
    private WebElement askChatbotAndWait(String message) {
        List<WebElement> existingMessages = driver.findElements(By.cssSelector("[data-testid='stChatMessage']"));
        int initialCount = existingMessages.size();

        WebElement chatInput = wait.until(ExpectedConditions.elementToBeClickable(
                By.cssSelector("textarea[data-testid='stChatInputTextArea'], textarea")
        ));
        chatInput.click();
        chatInput.clear();
        chatInput.sendKeys(message);
        chatInput.sendKeys(Keys.ENTER);

        // Fallback: click submit button if available
        try {
            List<WebElement> sendButtons = driver.findElements(By.cssSelector("button[data-testid='stChatInputSubmitButton']"));
            if (!sendButtons.isEmpty() && sendButtons.get(0).isEnabled()) {
                sendButtons.get(0).click();
            }
        } catch (Exception ignored) {}

        // Wait until both user message and full assistant reply (including markdown body) are rendered
        wait.until(driverInstance -> {
            List<WebElement> msgs = driverInstance.findElements(By.cssSelector("[data-testid='stChatMessage']"));
            if (msgs.size() >= initialCount + 2) {
                WebElement last = msgs.get(msgs.size() - 1);
                String text = last.getText().trim();
                // The category badge alone is ~40 chars; full response body is >100 chars
                return text.length() > 80 && !text.equals(message.trim());
            }
            return false;
        });

        List<WebElement> updatedMessages = driver.findElements(By.cssSelector("[data-testid='stChatMessage']"));
        return updatedMessages.get(updatedMessages.size() - 1);
    }
}

