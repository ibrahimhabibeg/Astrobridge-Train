import unittest

from src.evals.prompts import render_prompt, get_template
from src.evals.caption_tasks import get_caption_task
from src.evals.tasks import get_task


class TestPromptTemplates(unittest.TestCase):
    def test_caption_eval_distance_template(self):
        rendered = render_prompt(
            "caption_eval/distance.jinja2",
            options="A: z < 0.1\nB: 0.1 <= z <= 0.5\nC: z > 0.5",
            caption="This is a test spectrum caption.",
        )
        self.assertIn("You are an expert astrophysicist.", rendered)
        self.assertIn("Allowed categories:\nA: z < 0.1", rendered)
        self.assertIn("This is a test spectrum caption.", rendered)
        self.assertIn("FINAL ANSWER: [Letter]", rendered)

    def test_caption_eval_source_class_template(self):
        rendered = render_prompt(
            "caption_eval/source_class.jinja2",
            options="GALAXY, QSO, STAR",
            caption="This is a test galaxy caption.",
        )
        self.assertIn("astronomical source class into one of the following categories.", rendered)
        self.assertIn("GALAXY, QSO, STAR", rendered)
        self.assertIn("FINAL ANSWER: [Category]", rendered)

    def test_caption_eval_subclass_template(self):
        rendered = render_prompt(
            "caption_eval/subclass.jinja2",
            options="STARFORMING, AGN, STARBURST",
            caption="This is a test subclass caption.",
        )
        self.assertIn("astronomical source subclass into one of the following categories.", rendered)
        self.assertIn("STARFORMING, AGN, STARBURST", rendered)
        self.assertIn("FINAL ANSWER: [Category]", rendered)

    def test_caption_eval_emission_lines_template(self):
        rendered = render_prompt(
            "caption_eval/emission_lines.jinja2",
            candidate_lines="- Hα (6563 Å) [HALPHA]\n- [O III] 5007 Å [OIII_5007]",
            caption="Strong H-alpha and [O III] emission detected.",
        )
        self.assertIn("evaluate the following candidate checklist of emission lines", rendered)
        self.assertIn("- Hα (6563 Å) [HALPHA]", rendered)
        self.assertIn("Strong H-alpha and [O III] emission detected.", rendered)
        self.assertIn("EMISSION LINES: line1, line2, ...", rendered)
        self.assertIn("EMISSION LINES: NONE", rendered)

    def test_direct_eval_distance_templates(self):
        # Image mode
        img_prompt = render_prompt(
            "direct_eval/distance_image.jinja2",
            options="A: z < 0.1\nB: 0.1 <= z <= 0.5\nC: z > 0.5",
        )
        self.assertIn("classify the distance", img_prompt)
        self.assertIn("FINAL ANSWER: [Letter]", img_prompt)

        # Text mode
        txt_prompt = render_prompt(
            "direct_eval/distance_text.jinja2",
            options="A: z < 0.1\nB: 0.1 <= z <= 0.5\nC: z > 0.5",
            spectrum_text="Flux values: 1.2, 1.5, 2.0",
        )
        self.assertIn("Flux values: 1.2, 1.5, 2.0", txt_prompt)

        # Default mode
        def_prompt = render_prompt(
            "direct_eval/distance_default.jinja2",
            options="A: z < 0.1\nB: 0.1 <= z <= 0.5\nC: z > 0.5",
        )
        self.assertIn("Based on the spectrum provided", def_prompt)

    def test_direct_eval_source_templates(self):
        img_prompt = render_prompt(
            "direct_eval/source_class_image.jinja2",
            options="GALAXY, QSO, STAR",
        )
        self.assertIn("GALAXY, QSO, STAR", img_prompt)
        self.assertIn("FINAL ANSWER: [Category]", img_prompt)

        txt_prompt = render_prompt(
            "direct_eval/source_class_text.jinja2",
            options="GALAXY, QSO, STAR",
            spectrum_text="Wavelength: 4000-9000",
        )
        self.assertIn("Wavelength: 4000-9000", txt_prompt)

    def test_direct_eval_emission_lines_templates(self):
        img_prompt = render_prompt(
            "direct_eval/emission_lines_image.jinja2",
            candidate_lines="Hα, Hβ, [O III]",
        )
        self.assertIn("shown in the image", img_prompt)
        self.assertIn("Hα, Hβ, [O III]", img_prompt)
        self.assertIn("EMISSION LINES: line1, line2, ...", img_prompt)

        txt_prompt = render_prompt(
            "direct_eval/emission_lines_text.jinja2",
            candidate_lines="Hα, Hβ, [O III]",
            spectrum_text="Array: [1, 2, 3]",
        )
        self.assertIn("following spectrum data", txt_prompt)
        self.assertIn("Array: [1, 2, 3]", txt_prompt)

        def_prompt = render_prompt(
            "direct_eval/emission_lines_default.jinja2",
            candidate_lines="Hα, Hβ, [O III]",
        )
        self.assertIn("identify all visible emission lines", def_prompt)

    def test_caption_task_integration(self):
        # Distance task
        dist_task = get_caption_task("caption_distance")
        p1 = dist_task.build_frontier_prompt("Test caption for redshift.")
        self.assertIn("Allowed categories:", p1)
        self.assertIn("Test caption for redshift.", p1)

        # Source class task
        src_task = get_caption_task(
            "caption_source",
            active_classes={"GALAXY": "Galaxy", "QUASAR": "Quasar"},
        )
        p2 = src_task.build_frontier_prompt("Test caption for source class.")
        self.assertIn("source class", p2)
        self.assertIn("Galaxy, Quasar", p2)
        self.assertIn("Test caption for source class.", p2)

        # Subclass task
        sub_task = get_caption_task(
            "caption_subclass",
            active_classes={"STARFORMING": "Starforming", "AGN": "AGN"},
        )
        p3 = sub_task.build_frontier_prompt("Test caption for subclass.")
        self.assertIn("source subclass", p3)
        self.assertIn("Starforming, AGN", p3)
        self.assertIn("Test caption for subclass.", p3)

        # Emission lines task
        em_task = get_caption_task("caption_emission_lines")
        p4 = em_task.build_frontier_prompt(
            "Test caption with lines.",
            item={"candidate_query_lines": ["HALPHA", "HBETA"]},
        )
        self.assertIn("HALPHA", p4)
        self.assertIn("HBETA", p4)
        self.assertIn("EMISSION LINES: line1, line2, ...", p4)

    def test_tasks_integration(self):
        import pandas as pd

        dist_task = get_task("distance_classification")
        self.assertIn("classify the distance", dist_task.build_prompt(image_mode=True))
        self.assertIn("Wavelength data", dist_task.build_prompt(image_mode=False, spectrum_text="Wavelength data"))
        self.assertIn("Based on the spectrum provided", dist_task.build_prompt(image_mode=False))

        src_task = get_task(
            "source_classification",
            active_classes={"GALAXY": "GALAXY", "QSO": "QSO", "STAR": "STAR"},
        )
        self.assertIn("GALAXY, QSO, STAR", src_task.build_prompt(image_mode=True))

        dummy_gt = pd.DataFrame(
            [{"wiki_entity_id": "1", "LINE_NAME": "HALPHA", "SNR": 10.0}]
        )
        em_task = get_task("emission_lines", ground_truth_df=dummy_gt)
        self.assertIn("shown in the image", em_task.build_prompt(image_mode=True))
        self.assertIn("spectrum data", em_task.build_prompt(image_mode=False, spectrum_text="data"))

    def test_caption_generation_templates(self):
        ab_prompt = render_prompt("caption_generation/astrobridge.jinja2")
        self.assertIn("Describe this observation", ab_prompt)

        vis_prompt = render_prompt("caption_generation/vision_baseline.jinja2")
        self.assertIn("Provide a concise astronomical caption", vis_prompt)
        self.assertIn("Do not describe the plot layout", vis_prompt)

        txt_prompt = render_prompt("caption_generation/text_baseline.jinja2")
        self.assertIn("Provide a concise astronomical caption", txt_prompt)

        def_prompt = render_prompt("caption_generation/default.jinja2")
        self.assertIn("Describe the given spectrum in detail", def_prompt)

    def test_resolve_caption_prompt(self):
        from src.evals.caption_responders import (
            resolve_caption_prompt,
            DEFAULT_SPECTRUM_CAPTION_PROMPT,
            get_caption_responder,
        )

        # 1. Default fallback
        p_def = resolve_caption_prompt({})
        self.assertEqual(p_def, DEFAULT_SPECTRUM_CAPTION_PROMPT)

        # 2. Explicit text override
        p_custom = resolve_caption_prompt({"caption_prompt": "Custom user prompt."})
        self.assertEqual(p_custom, "Custom user prompt.")

        # 3. Via caption_prompt_template
        p_tmpl = resolve_caption_prompt(
            {"caption_prompt_template": "caption_generation/astrobridge.jinja2"}
        )
        self.assertIn("Describe this observation", p_tmpl)

        # 4. Via caption_prompt ending in .jinja2
        p_j2 = resolve_caption_prompt(
            {"caption_prompt": "caption_generation/astrobridge.jinja2"}
        )
        self.assertIn("Describe this observation", p_j2)

        # 5. Mock responder default initialization
        mock_resp = get_caption_responder({"responder_type": "mock"}, "cpu")
        self.assertEqual(mock_resp.caption_prompt, DEFAULT_SPECTRUM_CAPTION_PROMPT)


if __name__ == "__main__":
    unittest.main()

