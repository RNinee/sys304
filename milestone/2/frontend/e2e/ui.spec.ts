import { expect, test } from "@playwright/test";

test("welcome screen renders the classifier heading", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("welcome-title")).toHaveText(
    "Disaster tweet classifier",
  );
  await expect(page.getByLabel("Message input")).toBeVisible();
});

test("submitting a tweet calls /predict and shows the model reply", async ({
  page,
}) => {
  await page.route("**/predict", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        target: 1,
        label: "disaster",
        confidence: 0.91,
        probabilities: { disaster: 0.91, not_disaster: 0.09 },
        model: "qwen2.5-1.5b-lora",
        input: "Forest fire near La Ronge Sask. Canada",
      }),
    });
  });

  await page.goto("/");
  await page.getByLabel("Message input").fill(
    "Forest fire near La Ronge Sask. Canada",
  );
  await page.getByLabel("Send message").click();
  await expect(page.getByText(/real disaster/i)).toBeVisible();
  await expect(page.getByText(/qwen2\.5-1\.5b-lora/)).toBeVisible();
});
