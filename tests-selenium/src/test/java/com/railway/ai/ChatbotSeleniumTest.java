package com.railway.ai;

import io.github.bonigarcia.wdm.WebDriverManager;
import org.junit.jupiter.api.*;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * RailWay AI - Selenium WebDriver Automated Test Suite
 *
 * Validates real, live browser interactions against the running Streamlit chatbot:
 * 1. App loading and header presence.
 * 2. Sending a valid railway domain question (RAC/Waitlist) and verifying AI response.
 * 3. Sending an out-of-domain query (e.g. cake recipe) and verifying rejection safeguards.
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

        // Automatically configure ChromeDriver compatible with installed Google Chrome
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
    }

    @AfterAll
    public static void tearDown() {
        if (driver != null) {
            driver.quit();
            System.out.println("Selenium WebDriver closed successfully.");
        }
    }

    /**
     * Test 1: Verify the Chatbot Application loads and renders the title and sidebar
     */
    @Test
    @Order(1)
    @DisplayName("Verify Chatbot Page Loads and Displays Railway AI Header")
    public void testChatbotLoads() {
        driver.get(appUrl);

        // Wait for Streamlit root application container and chat input
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
     * Test 2: Send a genuine railway domain question (RAC inquiry) and assert AI response
     */
    @Test
    @Order(2)
    @DisplayName("Verify Railway Question on RAC Yields Domain-Accurate AI Response")
    public void testRailwayQuestion_RAC() throws InterruptedException {
        driver.get(appUrl);
        Thread.sleep(1500); // Allow Streamlit session to hydrate

        String query = "What is the rule for RAC ticket and do I get a berth?";
        sendChatMessage(query);

        // Wait for at least one assistant message response to appear
        WebElement assistantMessage = waitForLatestAssistantMessage();
        String responseText = assistantMessage.getText();
        System.out.println("Assistant Response to RAC query:\n" + responseText);

        assertNotNull(responseText, "Response text must not be null.");
        assertFalse(responseText.trim().isEmpty(), "Response text must not be empty.");

        // Validate that response contains railway/RAC domain knowledge
        boolean containsDomainKeyword = responseText.toLowerCase().contains("rac") ||
                responseText.toLowerCase().contains("berth") ||
                responseText.toLowerCase().contains("seat") ||
                responseText.toLowerCase().contains("reservation") ||
                responseText.toLowerCase().contains("passenger");

        assertTrue(containsDomainKeyword,
                "Expected response to mention RAC, berth, or seat details, but received: " + responseText);
    }

    /**
     * Test 3: Send an out-of-domain query (e.g. baking cake) and assert boundary rejection
     */
    @Test
    @Order(3)
    @DisplayName("Verify Unrelated Out-of-Domain Question is Rejected by Guardrails")
    public void testUnrelatedQuestion_Rejection() throws InterruptedException {
        driver.get(appUrl);
        Thread.sleep(1500);

        String unrelatedQuery = "How do I bake a chocolate cake with chocolate frosting?";
        sendChatMessage(unrelatedQuery);

        WebElement assistantMessage = waitForLatestAssistantMessage();
        String responseText = assistantMessage.getText();
        System.out.println("Assistant Response to Unrelated query:\n" + responseText);

        // Check that rejection safeguards or out-of-domain notification is triggered
        boolean isRejected = responseText.toLowerCase().contains("railway") ||
                responseText.toLowerCase().contains("irctc") ||
                responseText.toLowerCase().contains("scope") ||
                responseText.toLowerCase().contains("train") ||
                responseText.toLowerCase().contains("assist") ||
                responseText.toLowerCase().contains("specialize");

        assertTrue(isRejected,
                "Out-of-domain question should be guided back to Railway domain. Received: " + responseText);
    }

    /**
     * Helper to find the chat input, type text, and submit via Enter or Send Button
     */
    private void sendChatMessage(String message) {
        WebElement chatInput = wait.until(ExpectedConditions.elementToBeClickable(
                By.cssSelector("textarea[data-testid='stChatInputTextArea'], textarea")
        ));
        chatInput.click();
        chatInput.clear();
        chatInput.sendKeys(message);
        chatInput.sendKeys(Keys.ENTER);

        // Fallback: try clicking send button if present
        try {
            List<WebElement> sendButtons = driver.findElements(By.cssSelector("button[data-testid='stChatInputSubmitButton']"));
            if (!sendButtons.isEmpty() && sendButtons.get(0).isEnabled()) {
                sendButtons.get(0).click();
            }
        } catch (Exception ignored) {
        }
    }

    /**
     * Helper to wait for the latest assistant message bubble
     */
    private WebElement waitForLatestAssistantMessage() {
        // Wait for Streamlit assistant chat message container
        wait.until(ExpectedConditions.presenceOfElementLocated(
                By.cssSelector("[data-testid='stChatMessage']")
        ));

        // Allow up to 15 seconds for inference completion
        long endTime = System.currentTimeMillis() + 15000;
        WebElement targetAssistant = null;

        while (System.currentTimeMillis() < endTime) {
            List<WebElement> chatMessages = driver.findElements(By.cssSelector("[data-testid='stChatMessage']"));
            if (chatMessages.size() >= 2) {
                // The last message is the assistant's reply
                WebElement lastMsg = chatMessages.get(chatMessages.size() - 1);
                if (!lastMsg.getText().trim().isEmpty()) {
                    targetAssistant = lastMsg;
                    break;
                }
            }
            try {
                Thread.sleep(800);
            } catch (InterruptedException ignored) {}
        }

        assertNotNull(targetAssistant, "Assistant failed to generate response within timeout.");
        return targetAssistant;
    }
}
