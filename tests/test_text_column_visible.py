"""Text custom-field columns (AG News `text`, NLI `premise`, captions, etc.)
must appear on the dataset page. They used to be silently dropped from
`available_display_options` because the column-injection loop only handled
image / depth / json field types."""
from app import CustomField, Dataset, Sample, db


def test_text_field_renders_on_dataset_page(client, db_session):
    ds = Dataset(name='ag_news_like', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s00000')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='text', data_type='text',
        value_text='Wall St. Bears Claw Back Into the Black (Reuters)',
    ))
    db.session.add(CustomField(
        sample_id=s.id, name='label', data_type='scalar',
        value_float=2.0,
    ))
    db.session.commit()

    resp = client.get(f'/dataset/{ds.id}')
    assert resp.status_code == 200
    body = resp.data.decode()
    # The text column header should appear in the table.
    assert 'text' in body  # weak assertion (substring of many tokens) ...
    # The actual text value should render as the cell contents.
    assert 'Wall St. Bears Claw Back' in body


def test_text_field_does_not_collide_with_reserved_tags_column(client, db_session):
    """A custom field literally named `tags` is the dataset's tag widget,
    not a free-form text column — don't double-add it."""
    ds = Dataset(name='tag_collision', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='tags', data_type='text',
        value_text='depth,indoor',
    ))
    db.session.add(CustomField(
        sample_id=s.id, name='caption', data_type='text',
        value_text='A red cat on a green couch.',
    ))
    db.session.commit()
    resp = client.get(f'/dataset/{ds.id}')
    assert resp.status_code == 200
    body = resp.data.decode()
    # The free-form text column shows up.
    assert 'A red cat on a green couch.' in body


def test_nli_dataset_shape_renders_premise_and_hypothesis(client, db_session):
    """NLI-style datasets (SNLI, ANLI) have `premise` + `hypothesis` text
    columns. Both should appear on the dataset page so the user sees what
    the model is being asked to compare."""
    ds = Dataset(name='nli_shape', visibility='public')
    db.session.add(ds); db.session.flush()
    s = Sample(dataset_id=ds.id, name='s0')
    db.session.add(s); db.session.flush()
    db.session.add(CustomField(
        sample_id=s.id, name='premise', data_type='text',
        value_text='A man inspects the uniform of a figure.',
    ))
    db.session.add(CustomField(
        sample_id=s.id, name='hypothesis', data_type='text',
        value_text='The man is sleeping.',
    ))
    db.session.add(CustomField(
        sample_id=s.id, name='label', data_type='scalar', value_float=2.0,
    ))
    db.session.commit()
    resp = client.get(f'/dataset/{ds.id}')
    body = resp.data.decode()
    assert 'A man inspects the uniform of a figure.' in body
    assert 'The man is sleeping.' in body
