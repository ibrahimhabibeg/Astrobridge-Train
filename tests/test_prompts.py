import unittest

from src.evals.prompts import render_prompt
from src.evals.responders import (
    DEFAULT_SPECTRUM_CAPTION_PROMPT,
    resolve_caption_prompt,
    get_responder,
)
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

    def test_caption_eval_source_template(self):
        for tmpl in ["caption_eval/source.jinja2", "caption_eval/source_class.jinja2"]:
            rendered = render_prompt(
                tmpl,
                options="A: Galaxy\nB: Quasar",
                caption="This is a test galaxy caption.",
            )
            self.assertIn("astronomical source class", rendered)
            self.assertIn("A: Galaxy\nB: Quasar", rendered)
            self.assertIn("FINAL ANSWER: [Letter]", rendered)

    def test_caption_eval_subclass_template(self):
        rendered = render_prompt(
            "caption_eval/subclass.jinja2",
            options="A: AGN\nB: Starburst\nC: Starforming",
            caption="This is a test subclass caption.",
        )
        self.assertIn("astronomical source subclass", rendered)
        self.assertIn("A: AGN\nB: Starburst\nC: Starforming", rendered)
        self.assertIn("FINAL ANSWER: [Letter]", rendered)

    def test_caption_eval_emission_lines_template(self):
        rendered = render_prompt(
            "caption_eval/emission_lines.jinja2",
            candidate_lines="A: HALPHA\nB: OIII_5007",
            caption="Strong H-alpha and [O III] emission detected.",
        )
        self.assertIn("candidate checklist of emission lines", rendered)
        self.assertIn("A: HALPHA", rendered)
        self.assertIn("Strong H-alpha and [O III] emission detected.", rendered)
        self.assertIn("FINAL ANSWER: [Letter(s) separated by comma, e.g. A, B]", rendered)
        self.assertIn("FINAL ANSWER: NONE", rendered)

    def test_task_frontier_prompts(self):
        dist_task = get_task("distance")
        p1 = dist_task.build_frontier_prompt("Caption for distance")
        self.assertIn("Allowed categories:", p1)
        self.assertIn("Caption for distance", p1)

        src_task = get_task("source")
        p2 = src_task.build_frontier_prompt("Caption for source")
        self.assertIn("A: Galaxy", p2)
        self.assertIn("B: Quasar", p2)
        self.assertIn("Caption for source", p2)

        sub_task = get_task("subclass")
        p3 = sub_task.build_frontier_prompt("Caption for subclass")
        self.assertIn("A: AGN", p3)
        self.assertIn("B: Starburst", p3)
        self.assertIn("Caption for subclass", p3)

        em_task = get_task("emission_lines")
        p4 = em_task.build_frontier_prompt("Caption with emission lines", item={"candidate_query_lines": ["HALPHA", "HBETA"]})
        self.assertIn("HALPHA", p4)
        self.assertIn("HBETA", p4)

    def test_caption_generation_templates(self):
        ab = render_prompt("caption_generation/astrobridge.jinja2")
        self.assertIn("Describe this observation", ab)

        vis = render_prompt("caption_generation/vision_baseline.jinja2")
        self.assertIn("concise astronomical caption", vis)

        txt = render_prompt("caption_generation/text_baseline.jinja2")
        self.assertIn("concise astronomical caption", txt)

        def_p = render_prompt("caption_generation/default.jinja2")
        self.assertIn("Describe the given spectrum in detail", def_p)

    def test_resolve_caption_prompt(self):
        p_def = resolve_caption_prompt({})
        self.assertEqual(p_def, DEFAULT_SPECTRUM_CAPTION_PROMPT)

        p_custom = resolve_caption_prompt({"caption_prompt": "Custom user prompt."})
        self.assertEqual(p_custom, "Custom user prompt.")

        p_tmpl = resolve_caption_prompt({"caption_prompt_template": "caption_generation/astrobridge.jinja2"})
        self.assertIn("Describe this observation", p_tmpl)

        p_j2 = resolve_caption_prompt({"caption_prompt": "caption_generation/astrobridge.jinja2"})
        self.assertIn("Describe this observation", p_j2)

        mock_resp = get_responder({"responder_type": "mock"}, "cpu")
        self.assertEqual(mock_resp.caption_prompt, DEFAULT_SPECTRUM_CAPTION_PROMPT)


if __name__ == "__main__":
    unittest.main()
