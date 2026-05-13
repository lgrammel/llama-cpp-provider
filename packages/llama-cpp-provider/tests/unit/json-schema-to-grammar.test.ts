import { describe, it, expect } from "vitest";
import type { JSONSchema7 } from "@ai-sdk/provider";
import {
  convertJsonSchemaToGrammar,
  SchemaConverter,
} from "../../src/json-schema-to-grammar.js";

describe("convertJsonSchemaToGrammar", () => {
  describe("primitive types", () => {
    it("converts string type", () => {
      const schema: JSONSchema7 = { type: "string" };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain('"\\"" char*');
    });

    it("converts number type", () => {
      const schema: JSONSchema7 = { type: "number" };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("integral-part");
      expect(grammar).toContain("decimal-part");
    });

    it("converts integer type", () => {
      const schema: JSONSchema7 = { type: "integer" };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("integral-part");
    });

    it("converts boolean type", () => {
      const schema: JSONSchema7 = { type: "boolean" };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain('("true" | "false")');
    });

    it("converts null type", () => {
      const schema: JSONSchema7 = { type: "null" };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain('"null"');
    });
  });

  describe("object type", () => {
    it("converts simple object with properties", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          name: { type: "string" },
          age: { type: "integer" },
        },
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain('"{" space');
      expect(grammar).toContain('\\"name\\"');
      expect(grammar).toContain('\\"age\\"');
      expect(grammar).toContain('"}" space');
    });

    it("converts object with required properties", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          name: { type: "string" },
          age: { type: "integer" },
        },
        required: ["name"],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("name-kv");
      // Age should be optional
      expect(grammar).toContain("age-kv");
    });

    it("converts object with all properties required", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          name: { type: "string" },
          age: { type: "integer" },
        },
        required: ["name", "age"],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("name-kv");
      expect(grammar).toContain("age-kv");
    });

    it("converts nested object", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          person: {
            type: "object",
            properties: {
              name: { type: "string" },
            },
          },
        },
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("person");
      expect(grammar).toContain("person-name");
    });

    it("converts empty object schema", () => {
      const schema: JSONSchema7 = {};
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("object");
    });
  });

  describe("array type", () => {
    it("converts array of strings", () => {
      const schema: JSONSchema7 = {
        type: "array",
        items: { type: "string" },
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain('"[" space');
      expect(grammar).toContain('"]" space');
      // The item rule name should contain 'item' in some form
      expect(grammar).toMatch(/item|char/);
    });

    it("converts array with minItems", () => {
      const schema: JSONSchema7 = {
        type: "array",
        items: { type: "string" },
        minItems: 1,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain('"[" space');
    });

    it("converts array with maxItems", () => {
      const schema: JSONSchema7 = {
        type: "array",
        items: { type: "string" },
        maxItems: 5,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain('"[" space');
    });

    it("converts array with minItems and maxItems", () => {
      const schema: JSONSchema7 = {
        type: "array",
        items: { type: "integer" },
        minItems: 2,
        maxItems: 10,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
    });

    it("converts array of objects", () => {
      const schema: JSONSchema7 = {
        type: "array",
        items: {
          type: "object",
          properties: {
            id: { type: "integer" },
            name: { type: "string" },
          },
        },
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("item");
      expect(grammar).toContain("id");
      expect(grammar).toContain("name");
    });

    it("converts tuple array (prefixItems)", () => {
      const schema: JSONSchema7 = {
        type: "array",
        prefixItems: [
          { type: "string" },
          { type: "integer" },
          { type: "boolean" },
        ],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      // Tuples generate inline rules or reference primitive types
      expect(grammar).toContain('"[" space');
      expect(grammar).toContain('"]" space');
      expect(grammar).toContain('"," space');
    });
  });

  describe("enum and const", () => {
    it("converts string enum", () => {
      const schema: JSONSchema7 = {
        enum: ["red", "green", "blue"],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      // Enum values are JSON stringified and escaped
      expect(grammar).toContain('\\"red\\"');
      expect(grammar).toContain('\\"green\\"');
      expect(grammar).toContain('\\"blue\\"');
    });

    it("converts mixed enum", () => {
      const schema: JSONSchema7 = {
        enum: ["active", 1, true, null],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain('\\"active\\"');
      expect(grammar).toContain('"1"');
      expect(grammar).toContain('"true"');
      expect(grammar).toContain('"null"');
    });

    it("converts const value", () => {
      const schema: JSONSchema7 = {
        const: "fixed-value",
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain('\\"fixed-value\\"');
    });
  });

  describe("string constraints", () => {
    it("converts string with minLength", () => {
      const schema: JSONSchema7 = {
        type: "string",
        minLength: 3,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("char");
      expect(grammar).toContain("{3,");
    });

    it("converts string with maxLength", () => {
      const schema: JSONSchema7 = {
        type: "string",
        maxLength: 10,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("char");
      expect(grammar).toContain(",10}");
    });

    it("converts string with minLength and maxLength", () => {
      const schema: JSONSchema7 = {
        type: "string",
        minLength: 5,
        maxLength: 20,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("char");
      expect(grammar).toContain("{5,20}");
    });

    it("converts string with pattern", () => {
      const schema: JSONSchema7 = {
        type: "string",
        pattern: "^[a-z]+$",
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("[a-z]+");
    });
  });

  describe("integer constraints", () => {
    it("converts integer with minimum", () => {
      const schema: JSONSchema7 = {
        type: "integer",
        minimum: 0,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("[0]");
      expect(grammar).toContain("[1-9]");
    });

    it("converts integer with maximum", () => {
      const schema: JSONSchema7 = {
        type: "integer",
        maximum: 100,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
    });

    it("converts integer with minimum and maximum", () => {
      const schema: JSONSchema7 = {
        type: "integer",
        minimum: 1,
        maximum: 10,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
    });

    it("converts integer with exclusiveMinimum", () => {
      const schema: JSONSchema7 = {
        type: "integer",
        exclusiveMinimum: 0,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
    });

    it("converts integer with exclusiveMaximum", () => {
      const schema: JSONSchema7 = {
        type: "integer",
        exclusiveMaximum: 100,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
    });
  });

  describe("string formats", () => {
    it("converts date format", () => {
      const schema: JSONSchema7 = {
        type: "string",
        format: "date",
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("date");
    });

    it("converts time format", () => {
      const schema: JSONSchema7 = {
        type: "string",
        format: "time",
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("time");
    });

    it("converts date-time format", () => {
      const schema: JSONSchema7 = {
        type: "string",
        format: "date-time",
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("date-time");
    });

    it("converts uuid format", () => {
      const schema: JSONSchema7 = {
        type: "string",
        format: "uuid",
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("[0-9a-fA-F]");
    });
  });

  describe("composition keywords", () => {
    it("converts oneOf", () => {
      const schema: JSONSchema7 = {
        oneOf: [{ type: "string" }, { type: "integer" }],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("|");
    });

    it("converts anyOf", () => {
      const schema: JSONSchema7 = {
        anyOf: [{ type: "string" }, { type: "number" }, { type: "boolean" }],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("|");
    });

    it("converts allOf with object properties", () => {
      const schema: JSONSchema7 = {
        allOf: [
          {
            type: "object",
            properties: {
              name: { type: "string" },
            },
          },
          {
            type: "object",
            properties: {
              age: { type: "integer" },
            },
          },
        ],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("name");
      expect(grammar).toContain("age");
    });

    it("converts type array (union of types)", () => {
      const schema: JSONSchema7 = {
        type: ["string", "null"],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("|");
    });
  });

  describe("$ref resolution", () => {
    it("resolves local $ref to $defs", () => {
      const schema: JSONSchema7 = {
        $defs: {
          name: { type: "string" },
        },
        type: "object",
        properties: {
          firstName: { $ref: "#/$defs/name" },
          lastName: { $ref: "#/$defs/name" },
        },
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("firstName");
      expect(grammar).toContain("lastName");
    });

    it("resolves local $ref to definitions", () => {
      const schema: JSONSchema7 = {
        definitions: {
          address: {
            type: "object",
            properties: {
              street: { type: "string" },
              city: { type: "string" },
            },
          },
        },
        type: "object",
        properties: {
          homeAddress: { $ref: "#/definitions/address" },
        },
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("homeAddress");
    });
  });

  describe("additionalProperties", () => {
    it("converts object with additionalProperties: true", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          name: { type: "string" },
        },
        additionalProperties: true,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("additional");
    });

    it("converts object with additionalProperties schema", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          name: { type: "string" },
        },
        additionalProperties: { type: "integer" },
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("additional");
    });

    it("converts object with additionalProperties: false (implicit)", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          name: { type: "string" },
        },
        additionalProperties: false,
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).not.toContain("additional-kv");
    });
  });

  describe("complex schemas", () => {
    it("converts a recipe schema", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          name: { type: "string" },
          ingredients: {
            type: "array",
            items: {
              type: "object",
              properties: {
                name: { type: "string" },
                amount: { type: "string" },
              },
              required: ["name", "amount"],
            },
          },
          steps: {
            type: "array",
            items: { type: "string" },
          },
          prepTime: { type: "integer", minimum: 0 },
          cookTime: { type: "integer", minimum: 0 },
        },
        required: ["name", "ingredients", "steps"],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("name");
      expect(grammar).toContain("ingredients");
      expect(grammar).toContain("steps");
    });

    it("converts a user profile schema", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          id: { type: "string", format: "uuid" },
          email: { type: "string" },
          name: { type: "string", minLength: 1, maxLength: 100 },
          age: { type: "integer", minimum: 0, maximum: 150 },
          role: { enum: ["admin", "user", "guest"] },
          active: { type: "boolean" },
          tags: {
            type: "array",
            items: { type: "string" },
            minItems: 0,
            maxItems: 10,
          },
        },
        required: ["id", "email", "name"],
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("id");
      expect(grammar).toContain("email");
      expect(grammar).toContain("name");
      expect(grammar).toContain("age");
      expect(grammar).toContain("role");
      expect(grammar).toContain("active");
      expect(grammar).toContain("tags");
    });
  });

  describe("edge cases", () => {
    it("handles property names with special characters", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          "first-name": { type: "string" },
          last_name: { type: "string" },
        },
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("first-name");
      expect(grammar).toContain("last_name");
    });

    it("handles deeply nested objects", () => {
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          level1: {
            type: "object",
            properties: {
              level2: {
                type: "object",
                properties: {
                  level3: {
                    type: "object",
                    properties: {
                      value: { type: "string" },
                    },
                  },
                },
              },
            },
          },
        },
      };
      const grammar = convertJsonSchemaToGrammar(schema);

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("level1");
      expect(grammar).toContain("level2");
      expect(grammar).toContain("level3");
      expect(grammar).toContain("value");
    });

    it("throws error for unrecognized schema type", () => {
      const schema: JSONSchema7 = { type: "unknown" as any };

      expect(() => convertJsonSchemaToGrammar(schema)).toThrow(
        "Unrecognized schema"
      );
    });

    it("throws error for pattern not starting with ^", () => {
      const schema: JSONSchema7 = {
        type: "string",
        pattern: "[a-z]+$",
      };

      expect(() => convertJsonSchemaToGrammar(schema)).toThrow(
        'Pattern must start with "^" and end with "$"'
      );
    });

    it("throws error for pattern not ending with $", () => {
      const schema: JSONSchema7 = {
        type: "string",
        pattern: "^[a-z]+",
      };

      expect(() => convertJsonSchemaToGrammar(schema)).toThrow(
        'Pattern must start with "^" and end with "$"'
      );
    });
  });
});

describe("SchemaConverter", () => {
  describe("options", () => {
    it("respects propOrder option", () => {
      const converter = new SchemaConverter({
        propOrder: { age: 1, name: 2 },
      });
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          name: { type: "string" },
          age: { type: "integer" },
        },
      };

      converter.resolveRefs(schema);
      converter.visit(schema, "");
      const grammar = converter.formatGrammar();

      // With propOrder, age should come before name in the grammar
      const ageIndex = grammar.indexOf("age-kv");
      const nameIndex = grammar.indexOf("name-kv");
      expect(ageIndex).toBeLessThan(nameIndex);
    });

    it("respects dotall option for patterns", () => {
      const converter = new SchemaConverter({ dotall: true });
      const schema: JSONSchema7 = {
        type: "string",
        pattern: "^.+$",
      };

      converter.resolveRefs(schema);
      converter.visit(schema, "");
      const grammar = converter.formatGrammar();

      expect(grammar).toContain("dot");
      expect(grammar).toContain("U00000000");
    });
  });

  describe("formatGrammar", () => {
    it("produces sorted grammar rules", () => {
      const converter = new SchemaConverter();
      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          zebra: { type: "string" },
          apple: { type: "string" },
        },
      };

      converter.resolveRefs(schema);
      converter.visit(schema, "");
      const grammar = converter.formatGrammar();

      // Rules should be sorted alphabetically
      const lines = grammar.split("\n").filter((l) => l.includes("::="));
      const ruleNames = lines.map((l) => l.split("::=")[0].trim());

      const sortedNames = [...ruleNames].sort();
      expect(ruleNames).toEqual(sortedNames);
    });

    it("includes space rule", () => {
      const converter = new SchemaConverter();
      const schema: JSONSchema7 = { type: "string" };

      converter.resolveRefs(schema);
      converter.visit(schema, "");
      const grammar = converter.formatGrammar();

      expect(grammar).toContain("space ::=");
    });
  });

  describe("resolveRefs", () => {
    it("resolves nested $ref correctly", () => {
      const converter = new SchemaConverter();
      const schema: JSONSchema7 = {
        $defs: {
          inner: { type: "string" },
        },
        type: "object",
        properties: {
          value: { $ref: "#/$defs/inner" },
        },
      };

      converter.resolveRefs(schema);
      converter.visit(schema, "");
      const grammar = converter.formatGrammar();

      expect(grammar).toContain("root ::=");
      expect(grammar).toContain("value");
    });

    it("throws for unsupported remote $ref", () => {
      const converter = new SchemaConverter();
      const schema: JSONSchema7 = {
        $ref: "https://example.com/schema.json",
      };

      expect(() => converter.resolveRefs(schema)).toThrow(
        "Fetching remote schemas is not supported"
      );
    });

    it("throws for unsupported $ref format", () => {
      const converter = new SchemaConverter();
      const schema: JSONSchema7 = {
        $ref: "some-file.json",
      };

      expect(() => converter.resolveRefs(schema)).toThrow("Unsupported ref");
    });
  });
});
